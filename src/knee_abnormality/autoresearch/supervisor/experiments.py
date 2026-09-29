import ast
import difflib
import math
import subprocess
import sys
from pathlib import Path

import yaml

from knee_abnormality.autoresearch.campaign.binding import (
    candidate_binding,
    digest,
    external_evidence,
    launch_blockers,
    parent_changed,
    reference_seed,
    reference_snapshot,
)
from knee_abnormality.autoresearch.campaign.gates import (
    chained_counts,
    config_differences,
    run_limits,
    validate_config,
    without_fields,
)
from knee_abnormality.autoresearch.campaign.reports import DeveloperReport
from knee_abnormality.autoresearch.campaign.rules import (
    ACCEPTANCE_RULE,
    CALIBRATION_EPOCHS,
)
from knee_abnormality.autoresearch.runtime.isolation import (
    changed_files,
    check_scope,
    manifest,
    sandbox,
    snapshot,
)
from knee_abnormality.autoresearch.runtime.layout import (
    PREDICTIONS,
    campaign_fold,
    comparison_path,
    fold_dir,
)
from knee_abnormality.autoresearch.runtime.storage import (
    atomic_write,
    read_json,
    sha256,
    write_json,
)
from knee_abnormality.autoresearch.supervisor.common import (
    identifier,
)


class ExperimentMixin:
    def claim(self, repair=None):
        idea_id = self.state["queue"][0]
        incumbent = self.state["incumbent"]
        exp_id = identifier(idea_id)
        directory = self.output / "experiments" / exp_id
        source = Path(repair["workspace"] if repair else incumbent["source"])
        workspace = directory / "workspace"
        snapshot(source, workspace)
        exp = {
            "id": exp_id,
            "idea": idea_id,
            "directory": str(directory),
            "workspace": str(workspace),
            "source": str(directory / "source"),
            "implementation_parent": repair["implementation_parent"]
            if repair
            else incumbent["id"],
            "implementation_reference": repair["implementation_reference"]
            if repair
            else reference_snapshot(incumbent, campaign_fold(self.state)),
            "parent_source": repair["parent_source"] if repair else incumbent["source"],
            "repair_of": repair["id"] if repair else None,
            "repair_count": repair["repair_count"] + 1 if repair else 0,
            "stage": "needs_developer",
            "launched": False,
        }
        self.state["experiments"][exp_id] = exp
        self.state["claimed"] = exp_id
        idea = self.state["ideas"][idea_id]
        idea["status"] = "developing"
        idea.setdefault("experiments", []).append(exp_id)
        self.save()

    def prepare(self, exp, report):
        parsed = DeveloperReport.model_validate(report)
        if parsed.status != "prepared":
            raise ValueError("Cannot prepare a blocked developer report")
        workspace = Path(exp["workspace"])
        parent = Path(exp["parent_source"])
        after, changed = check_scope(workspace, manifest(parent), self.policy)
        for path in changed:
            if path.endswith(".py") and (workspace / path).exists():
                ast.parse((workspace / path).read_text(), filename=path)
        source_change = any(path.startswith("src/") for path in changed)
        if self.policy.require_source_change and not source_change:
            raise ValueError(
                "Candidate has no source change; scalar sweeps are outside this campaign"
            )
        if (
            self.policy.configuration_only_ideas
            and self.state["ideas"][exp["idea"]].get("source_change") is True
            and not source_change
        ):
            raise ValueError(
                "The idea declares a source change, but the candidate changes only configuration"
            )
        source = Path(exp["source"])
        snapshot(workspace, source)
        write_json(Path(exp["directory"]) / "manifest.json", after)
        diff = []
        for path in changed:
            old = (
                (parent / path).read_text().splitlines(True)
                if (parent / path).exists()
                else []
            )
            new = (
                (source / path).read_text().splitlines(True)
                if (source / path).exists()
                else []
            )
            diff.extend(
                difflib.unified_diff(
                    old, new, fromfile=f"parent/{path}", tofile=f"candidate/{path}"
                )
            )
        atomic_write(Path(exp["directory"]) / "changes.diff", "".join(diff))
        exp["developer_report"] = report
        exp["integration"] = parsed.integration.model_dump(mode="json")
        exp["activation"] = parsed.activation.model_dump(mode="json")
        self.config_gate(exp)

    def config_gate(self, exp):
        source = Path(exp["source"])
        exp["stage"] = "config_gate"
        self.submit(
            "config",
            source,
            sandbox(
                [
                    sys.executable,
                    "-m",
                    "knee_abnormality.config",
                    str(source / "configs/training.yaml"),
                ],
                source,
                [],
            ),
            payload={"experiment": exp["id"]},
            timeout=120,
        )

    def reconcile(self, previous):
        parent = Path(previous["parent_source"])
        if not parent.is_dir():
            raise ValueError("Recorded old parent source is missing")
        if previous.get("implementation_reference") and (
            digest(manifest(parent))
            != previous["implementation_reference"]["source_sha256"]
        ):
            raise ValueError(
                "Recorded old parent source changed; cannot reconcile safely"
            )
        candidate = Path(previous["source"])
        if candidate.exists() and manifest(candidate) != read_json(
            Path(previous["directory"]) / "manifest.json"
        ):
            raise ValueError(
                "Recorded candidate source changed; cannot reconcile safely"
            )
        if not candidate.exists():
            candidate = Path(previous["workspace"])
        if not candidate.is_dir():
            raise ValueError("Candidate source and workspace are missing")
        check_scope(candidate, manifest(parent), self.policy)
        changes = changed_files(manifest(parent), manifest(candidate))
        incumbent = self.state["incumbent"]
        self.claim()
        exp = self.state["experiments"][self.state["claimed"]]
        exp["reconciliation"] = {
            "candidate": previous["id"],
            "old_parent": previous["implementation_parent"],
            "new_parent": incumbent["id"],
            "conflicts": [],
            "previous_integration": previous.get("integration"),
            "reason": "Reassess integration against the current incumbent before renewed review",
        }
        previous["stage"] = "superseded_by_reconciliation"
        workspace = Path(exp["workspace"])
        merge_root = Path(exp["directory"]) / "reconciliation"
        merge_root.mkdir()
        for index, path in enumerate(changes):
            base_path, candidate_path, current_path = (
                parent / path,
                candidate / path,
                workspace / path,
            )
            base = base_path.read_bytes() if base_path.exists() else None
            proposed = candidate_path.read_bytes() if candidate_path.exists() else None
            current = current_path.read_bytes() if current_path.exists() else None
            if proposed == current:
                continue
            if current == base:
                if proposed is None:
                    current_path.unlink(missing_ok=True)
                else:
                    current_path.parent.mkdir(parents=True, exist_ok=True)
                    current_path.write_bytes(proposed)
                continue
            if proposed is None or current is None:
                exp["reconciliation"]["conflicts"].append(path)
                continue
            inputs = []
            for name, value in (
                ("current", current),
                ("base", base or b""),
                ("candidate", proposed),
            ):
                target = merge_root / f"{index}-{name}"
                target.write_bytes(value)
                inputs.append(str(target))
            result = subprocess.run(
                ["git", "merge-file", "--diff3", "--stdout", *inputs],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode < 0 or result.returncode > 127:
                raise ValueError(
                    f"Cannot reconcile {path}: {result.stderr.decode(errors='replace')}"
                )
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.write_bytes(result.stdout)
            if result.returncode:
                exp["reconciliation"]["conflicts"].append(path)
        write_json(merge_root / "record.json", exp["reconciliation"])
        exp["stage"] = "needs_developer"
        self.save()

    def seal(self):
        directory = self.output / "baseline"
        ledger = read_json(directory / "triplet_accounting.json")
        if not ledger["complete"] or not ledger["changed_training_studies"]:
            raise ValueError(
                "Baseline accounting is incomplete or training selection never changes"
            )
        exceptions = self.state["limits"]["reference_identity_exceptions"]
        expected = without_fields(
            read_json(
                fold_dir(self.state["reference"], campaign_fold(self.state))
                / "input_identity.json"
            ),
            exceptions,
        )
        actual = without_fields(
            read_json(directory / "input_identity.json"), exceptions, required=False
        )
        if actual != expected:
            raise ValueError(
                "Baseline data/preprocessing identity differs from historical control"
            )
        limits = self.state["limits"]
        if len(ledger["train"]) != min(CALIBRATION_EPOCHS, limits["epochs"]):
            raise ValueError("Baseline did not replay the calibration epochs")
        first = ledger["train"][0]
        if any(
            (row["visits"], row["triplets"]) != (first["visits"], first["triplets"])
            for row in ledger["train"]
        ):
            raise ValueError(
                "Baseline per-epoch training counts vary; totals cannot be sealed from a partial replay"
            )
        epoch_steps = math.ceil(
            math.ceil(first["visits"] / self.state["config"]["trainer"]["batch_size"])
            / self.state["config"]["trainer"]["accumulate_grad_batches"]
        )
        if epoch_steps * limits["epochs"] != limits["optimizer_steps"]:
            raise ValueError(
                "Historical optimizer steps differ from the sealed schedule"
            )
        cohorts = {
            "training_cohort_sha256": first["cohort_sha256"],
            "validation_cohort_sha256": ledger["validation"][0]["cohort_sha256"],
        }
        if self.state["budget"]["mode"] == "sealed":
            limits.update(
                training_triplets=first["triplets"] * limits["epochs"],
                training_visits=first["visits"] * limits["epochs"],
                **cohorts,
                validation_triplets=ledger["validation"][0]["triplets"],
            )
        else:
            limits.update(**cohorts)
        if self.chain:
            chained_counts(self.root, self.chain, limits, self.state["budget"])
        write_json(self.output / "limits.json", limits)
        self.state["sealed_sha256"] = sha256(self.output / "limits.json")
        self.state["status"] = "running"

    def review_binding(self, exp):
        if parent_changed(self.state, exp):
            raise ValueError(
                "Candidate implementation parent differs from the incumbent"
            )
        seed = reference_seed(self.state["incumbent"])
        exp["config"] = {**exp["config"], "seed": seed}
        exp["seed"] = seed
        return candidate_binding(
            self.state,
            exp,
            external_evidence(self.root, self.campaign, campaign_fold(self.state)),
        )

    def training(self, exp):
        blockers = launch_blockers(self.state, exp)
        if blockers:
            raise ValueError("Cannot launch: " + "; ".join(blockers))
        binding = self.review_binding(exp)
        approval = exp.get("implementation_review", {})
        if approval.get("decision") != "approve" or approval.get("binding") != binding:
            self.request_review(
                exp,
                refresh_reason="Approval changed before launch; inspect the updated evidence",
            )
            self.save()
            return
        source = Path(exp["source"])
        if manifest(source) != read_json(Path(exp["directory"]) / "manifest.json"):
            raise ValueError("Reviewed candidate source changed")
        if sha256(self.output / "limits.json") != self.state["sealed_sha256"]:
            raise ValueError("Sealed campaign limits changed")
        reference = self.state["incumbent"]
        reference_predictions = sha256(
            fold_dir(reference["run"], campaign_fold(self.state)) / PREDICTIONS
        )
        if reference_predictions != reference["predictions_sha256"]:
            raise ValueError("Incumbent predictions changed")
        config = {
            **exp["config"],
            "seed": binding["training_seed"],
            "name": exp["id"],
            "paths": {**exp["config"]["paths"], "output_root": exp["directory"]},
        }
        validate_config(self.state["config"], config, self.policy, self.state["budget"])
        epochs = config["trainer"]["max_epochs"]
        limits_path = Path(exp["directory"]) / "run-limits.json"
        write_json(limits_path, run_limits(self.state, epochs))
        exp.update(
            reference=dict(reference),
            seed=config["seed"],
            stage="launching",
            launch_binding=binding,
            active_idea_ids=binding["active_idea_ids"],
        )
        exp["reference_predictions_sha256"] = reference_predictions
        exp["comparison_keys"] = config_differences(
            read_json(Path(reference["run"]) / "effective_config.json"), config
        )
        self.state["running"] = exp["id"]
        self.submit_training(
            "training",
            config,
            Path(exp["directory"]),
            source,
            {"experiment": exp["id"]},
            self.state["limits"]["max_run_seconds"] - exp["preparation_seconds"],
            limits_path,
        )

    def submit_training(
        self, kind, config, directory, source, payload, timeout, limits_path
    ):
        config_path = directory / "config.yaml"
        atomic_write(config_path, yaml.safe_dump(config, sort_keys=False))
        writable = [directory]
        cache = Path.home() / ".cache"
        if cache.exists():
            writable.append(cache)
        self.submit(
            kind,
            source,
            sandbox(
                [
                    sys.executable,
                    "-m",
                    "scripts.pipeline.train",
                    str(config_path),
                    "--validation.folds",
                    f"[{campaign_fold(self.state)}]",
                    "--paths.output-root",
                    str(directory),
                    "--run-id",
                    "training",
                ],
                source,
                writable,
                gpu=True,
                readonly=[source, config_path, limits_path],
            ),
            payload=payload,
            timeout=timeout,
            gpu=self.state["gpu"],
            extra_env={"KNEE_AUTORESEARCH_LIMITS": str(limits_path)},
        )

    def dequeue_launched(self, exp):
        if (
            exp["launched"]
            or not (Path(exp["directory"]) / "training/execution.json").exists()
        ):
            return
        if self.state["queue"][0] != exp["idea"] or self.state["claimed"] != exp["id"]:
            raise ValueError("Launch no longer owns the reserved queue head")
        self.state["queue"].pop(0)
        self.state["claimed"] = None
        self.state["ideas"][exp["idea"]]["status"] = "launched"
        self.state["evidence_revision"] += 1
        exp.update(launched=True, stage="running")
        self.save()

    def comparison(self, exp):
        base = Path(self.state["base"])
        directory = Path(exp["directory"])
        reference = Path(exp["reference"]["run"])
        if (
            sha256(fold_dir(reference, campaign_fold(self.state)) / PREDICTIONS)
            != exp["reference_predictions_sha256"]
        ):
            raise ValueError("Launch-time reference predictions changed")
        if sha256(Path(self.state["headers"])) != self.state["headers_sha256"]:
            raise ValueError("Acquisition metadata changed")
        command = [
            sys.executable,
            "-m",
            "scripts.tools.compare_fold_runs",
            "--reference",
            str(reference),
            "--candidate",
            str(directory / "training"),
            "--output-dir",
            str(directory / "comparison"),
            "--fold",
            str(campaign_fold(self.state)),
            "--headers",
            self.state["headers"],
            "--bootstrap-draws",
            str(self.state["limits"]["bootstrap_draws"]),
            "--bootstrap-seed",
            str(self.state["limits"]["bootstrap_seed"]),
            "--allowed-config-differences",
            ",".join(exp["comparison_keys"]),
        ]
        identity_keys = self.policy.input_identity_keys
        if identity_keys:
            command += [
                "--allowed-input-identity-differences",
                ",".join(identity_keys),
            ]
        exceptions = (
            list(self.state["limits"]["reference_identity_exceptions"])
            if reference.resolve() == Path(self.state["reference"]).resolve()
            else []
        )
        exceptions += [key for key in identity_keys if key not in exceptions]
        if exceptions:
            command += ["--reference-identity-exceptions", ",".join(exceptions)]
        remaining = (
            self.state["limits"]["max_run_seconds"]
            - exp["preparation_seconds"]
            - exp["training_receipt"]["seconds"]
        )
        if remaining <= 0:
            self.completion(
                exp,
                "invalid",
                {"reasons": ["No evaluation time remains within the original ceiling"]},
            )
            self.save()
            return
        exp["stage"] = "comparing"
        command_file = directory / "comparison-command.json"
        write_json(
            command_file,
            {
                "source": str(base),
                "command": command,
                "comparison": str(comparison_path(exp)),
            },
        )
        self.submit(
            "comparison",
            self.root,
            sandbox(
                [
                    sys.executable,
                    "-m",
                    "scripts.tools.autoresearch",
                    "compare",
                    "--config",
                    str(command_file),
                    "--root",
                    str(self.root),
                ],
                self.root,
                [directory],
                readonly=[directory / "source", directory / "training"],
            ),
            payload={"experiment": exp["id"]},
            timeout=remaining,
        )

    def record_evaluation(self, exp, decision, evidence):
        evidence = {
            **evidence,
            "run_checks": exp.get("run_checks", {}),
            "acceptance_rule": dict(ACCEPTANCE_RULE),
        }
        exp.update(eligible_decision=decision, evidence=evidence)
        write_json(
            Path(exp["directory"]) / "evaluation.json",
            {"eligible_decision": decision, **evidence},
        )
        return evidence

    def completion(self, exp, decision, evidence):
        evidence = self.record_evaluation(exp, decision, evidence)
        exp["stage"] = "awaiting_decision"
        self.event(
            "completion",
            {
                "experiment": exp["id"],
                "eligible_decision": decision,
                "evidence": evidence,
            },
        )
