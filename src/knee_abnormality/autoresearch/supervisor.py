import ast
import copy
import difflib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

from knee_abnormality.autoresearch import providers
from knee_abnormality.autoresearch.campaigns import CampaignPolicy, load_campaign
from knee_abnormality.autoresearch.completion import completion_blockers
from knee_abnormality.autoresearch.context import build_context, prompt_sizes
from knee_abnormality.autoresearch.diagnostics import diagnostics_path, read_diagnostics
from knee_abnormality.autoresearch.gates import (
    ACCEPTANCE_RULE,
    active_incumbent_ideas,
    config_differences,
    dependency_blockers,
    evaluate,
    initial_limits,
    validate_config,
    verdict,
)
from knee_abnormality.autoresearch.integration import (
    candidate_binding,
    comparison_identity_keys,
    digest,
    external_evidence,
    launch_blockers,
    parent_changed,
    reference_seed,
    reference_snapshot,
)
from knee_abnormality.autoresearch.isolation import (
    changed_files,
    check_scope,
    environment,
    manifest,
    sandbox,
    snapshot,
)
from knee_abnormality.autoresearch.models import (
    REPORTS,
    DeveloperReport,
    OrchestratorReport,
    ResearchReport,
)
from knee_abnormality.autoresearch.storage import (
    alive,
    atomic_write,
    lock,
    process_identity,
    read_json,
    sha256,
    terminate,
    write_json,
)
from knee_abnormality.config import load_config


def identifier(prefix):
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def gpu_available(gpu: str):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect selected GPU: {result.stderr.strip()}")
    return not result.stdout.strip()


class Supervisor:
    def __init__(self, root: Path, campaign: str):
        self.root = root
        self.docs = root / "autoresearch/campaigns" / campaign
        definition = load_campaign(self.docs / "campaign.yaml")
        self.campaign = definition.model_dump(mode="json")
        self.policy = definition.policy
        if self.campaign["id"] != campaign:
            raise ValueError(f"Mismatched campaign: {campaign}")
        self.output = (root / self.campaign["artifacts_root"]).resolve()
        if not self.output.is_relative_to(root / "outputs/autoresearch"):
            raise ValueError("Campaign artifacts must stay in outputs/autoresearch")
        self.state_path = self.output / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else None
        if self.state is not None:
            if self.state["campaign"] != campaign:
                raise ValueError("Artifact directory belongs to a different campaign")
            sealed = CampaignPolicy.model_validate(self.state["policy"])
            if sealed != self.policy:
                raise ValueError(
                    "Campaign policy differs from its seal; create a new campaign"
                )
            self.policy = sealed
        self.wakeup = threading.Event()

    def save(self):
        self.state["revision"] += 1
        self.state["updated_at"] = time.time()
        write_json(self.state_path, self.state)
        self.project()

    def project(self):
        revision = self.state["revision"]
        atomic_write(
            self.docs / "backlog.yaml",
            yaml.safe_dump(
                {
                    "revision": revision,
                    "queue": self.state["queue"],
                },
                sort_keys=False,
            ),
        )
        atomic_write(
            self.docs / "ideas.yaml",
            yaml.safe_dump(
                {
                    "revision": revision,
                    "ideas": list(self.state["ideas"].values()),
                },
                sort_keys=False,
            ),
        )
        findings = "\n\n".join(
            f"## {entry['event']}\n\n{entry['finding']}\n\n{entry['rationale']}"
            + (
                "\n\nUnresolved concerns:\n"
                + "\n".join(f"- {value}" for value in entry["concerns"])
                if entry.get("concerns")
                else ""
            )
            + (
                "\n\nEvidence:\n"
                + "\n".join(f"- {value}" for value in entry["evidence_refs"])
                if entry.get("evidence_refs")
                else ""
            )
            for entry in self.state["findings"]
        )
        atomic_write(self.docs / "findings.md", f"# Campaign findings\n\n{findings}\n")

    def initialize(self, provider: str, gpu: str):
        if self.state is not None:
            raise ValueError("Campaign already exists; use resume")
        if (
            yaml.safe_load((self.docs / "ideas.yaml").read_text())["ideas"]
            or yaml.safe_load((self.docs / "backlog.yaml").read_text())["queue"]
        ):
            raise ValueError(
                "Research records exist without runtime state; restore state instead of resetting history"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        base = self.output / "baseline/source"
        if base.exists():
            raise ValueError(
                "Incomplete initialization exists; inspect baseline/ before starting again"
            )
        recipe = load_config(self.root / self.campaign["recipe"], folds=[0])
        config = recipe.model_dump(mode="json")
        reference = (self.root / self.campaign["reference_run"]).resolve()
        limits = initial_limits(reference, config)
        validate_config(config, config, self.policy)
        headers = (
            self.root
            / "outputs/analysis/sharing-disagreement-20260914/study_headers.parquet"
        )
        if not headers.is_file():
            raise ValueError(f"Missing bootstrap acquisition metadata: {headers}")
        snapshot(self.root, base)
        config["seed"] = reference_seed({"run": str(reference)})
        atomic_write(
            base / "configs/training.yaml", yaml.safe_dump(config, sort_keys=False)
        )
        write_json(self.output / "baseline/manifest.json", manifest(base))
        write_json(self.output / "baseline/config.json", config)
        write_json(self.output / "limits.json", limits)
        self.state = {
            "version": 2,
            "revision": 0,
            "campaign": self.campaign["id"],
            "policy": self.policy.model_dump(mode="json"),
            "provider": provider,
            "gpu": gpu,
            "status": "calibrating",
            "reason": None,
            "profiles": yaml.safe_load(
                (self.root / "autoresearch/agents.yaml").read_text()
            ),
            "base": str(base),
            "reference": str(reference),
            "headers": str(headers),
            "headers_sha256": sha256(headers),
            "limits": limits,
            "config": config,
            "incumbent": {
                "id": "historical-control",
                "source": str(base),
                "run": str(reference),
                "predictions_sha256": limits["reference_predictions_sha256"],
            },
            "queue": [],
            "ideas": {},
            "experiments": {},
            "jobs": {},
            "events": [],
            "findings": [],
            "claimed": None,
            "running": None,
            "evidence_revision": 0,
            "last_refill_revision": -1,
            "refill_below": self.campaign["backlog"]["refill_below"],
            "refill_to": self.campaign["backlog"]["refill_to"],
        }
        seeds = self.docs / "seed-ideas.yaml"
        if seeds.exists():
            self.import_ideas(seeds)
        else:
            self.save()

    def event(self, kind, payload):
        self.state["events"].append({"id": identifier(kind), "kind": kind, **payload})

    def import_ideas(self, path: Path):
        report = ResearchReport.model_validate(yaml.safe_load(path.read_text()))
        ids = [idea.id for idea in report.ideas]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Import requires nonempty ideas with unique IDs")
        known = set(self.state["ideas"])
        for event in self.state["events"]:
            known.update(
                idea["id"] for idea in event.get("report", {}).get("ideas", [])
            )
        if known.intersection(ids):
            raise ValueError(
                "Imported IDs already exist in the registry or pending proposals"
            )
        artifact = self.output / "imports" / identifier("ideas") / "report.json"
        value = report.model_dump(mode="json")
        write_json(artifact, value)
        self.event(
            "proposals",
            {
                "report": value,
                "artifact": str(artifact),
                "source": str(path.resolve()),
                "instructions": "These are existing repository ideas imported at the user's request. Review and rank eligible ideas before researching more. Preserve provenance and campaign ownership; do not queue blocked or previously rejected work. Deduplicate subsequent proposals against these mechanisms.",
            },
        )
        self.save()

    def active(self, kind):
        return any(
            job["kind"] == kind and not job["processed"]
            for job in self.state["jobs"].values()
        )

    def outstanding(self):
        return [
            job
            for job in self.state["jobs"].values()
            if not job["processed"] and not job.get("application_error")
        ]

    def create_job(
        self,
        kind,
        source: Path,
        command,
        *,
        payload=None,
        timeout=3600,
        prompt="",
        extra_env=None,
        gpu="",
    ):
        job_id = identifier(kind)
        directory = self.output / "jobs" / job_id
        directory.mkdir(parents=True)
        spec = {
            "source": str(source),
            "command": command,
            "timeout_seconds": timeout,
            "environment": extra_env or {},
            "gpu": gpu,
        }
        if kind == "training":
            spec["gpu_lock"] = str(
                self.root / "outputs/autoresearch/.gpu-locks" / f"{gpu}.lock"
            )
        atomic_write(directory / "stdin.txt", prompt)
        write_json(directory / "spec.json", spec)
        self.state["jobs"][job_id] = {
            "id": job_id,
            "kind": kind,
            "path": str(directory),
            "processed": False,
            "payload": payload or {},
            "created_at": time.time(),
        }
        return job_id, directory

    def launch_worker(self, job):
        directory = Path(job["path"])
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "knee_abnormality.autoresearch.cli",
                    "job",
                    "--job-dir",
                    str(directory),
                ],
                cwd=self.root,
                env=environment(self.root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        write_json(directory / "launcher.json", process_identity(process.pid))

    def submit(self, *args, **kwargs):
        job_id, directory = self.create_job(*args, **kwargs)
        self.save()
        self.launch_worker(self.state["jobs"][job_id])
        return job_id, directory

    def role(self, role, source: Path, context, *, payload=None, repair=False):
        if role != "completion_reporter":
            context = {
                **context,
                "external_evidence": external_evidence(self.root, self.campaign),
            }
        job_id, directory = self.create_job(role, source, [], payload=payload)
        if role == "completion_reporter":
            event = payload["event"]
            event["completion_reporting"] = {"status": "assigned", "job": job_id}
            self.state["experiments"][event["experiment"]]["completion_reporting"] = (
                dict(event["completion_reporting"])
            )
        profile = dict(self.state["profiles"][self.state["provider"]][role])
        if repair and self.state["provider"] == "codex":
            profile = {"model": "gpt-5.6-sol", "effort": "high"}
        command = providers.invocation(
            self.state["provider"],
            role,
            profile,
            source,
            directory,
            self.policy,
        )
        brief = (
            self.root / "autoresearch/roles" / f"{role.replace('_', '-')}.md"
        ).read_text()
        contract = (
            Path(self.state["base"])
            / "autoresearch/campaigns"
            / self.state["campaign"]
            / "contract.md"
        ).read_text()
        context = build_context(
            self.state, role, context, directory, self.docs, self.state_path
        )
        context["python_executable"] = sys.executable
        context["incumbent_training_seed"] = reference_seed(self.state["incumbent"])
        context["acceptance_rule"] = dict(ACCEPTANCE_RULE)
        if role != "completion_reporter":
            context["campaign_policy"] = self.policy.model_dump(
                mode="json",
                exclude={"developer_instructions", "researcher_instructions"},
            )
        guidance = (Path(self.state["base"]) / "AGENTS.md").read_text()
        evidence_guidance = (
            "Use only the supplied completion facts and their original artifacts. "
            "Do not research, review code, change priorities or infer scientific conclusions. "
            "Return a factual report or escalate uncertainty; no decision fields are permitted. "
            if role == "completion_reporter"
            else "Full history and evidence remain available through history_index and artifact paths. "
            "Retrieve relevant originals; absent facts are unknown. "
        )
        prompt = (
            f"{guidance}\n\n{brief}\n\n{contract}\n\n"
            "User-directed seed policy supersedes fresh-seed instructions in archived contracts and findings: "
            "every new candidate uses the current incumbent's recorded training seed, enforced by the supervisor. "
            "Historical runs retain their actual seeds; do not relabel them as matched. "
            "The live state's incumbent and recorded user reference transition supersede starting-reference instructions in archived contracts. "
            "User-directed adoption policy supersedes interval-sign gates in archived contracts and findings: "
            "adopt only when valid macro AUC improvement >= 0.001 AND the central 80% paired-bootstrap lower bound is strictly positive; otherwise discard. "
            "The 95% and per-label intervals remain diagnostic; do not add other adoption thresholds. "
            "Do not keep pending implementation branches or schedule automatic confirmation runs. "
            "Campaign authorization overrides ordinary user-only training guidance, but ONLY the supervisor launches training. "
            "Do not launch training, use GPUs, spawn subagents, write/run tests, or poll jobs. "
            "NO SUBMISSIONS OR PUBLICATION: only local training and local validation are authorized. "
            "Never submit to Kaggle, upload models/predictions, publish datasets/notebooks, or launch remote jobs. "
            "Reading public research/documentation is allowed. "
            "All canonical docs are read-only to this invocation: return the requested structured report; "
            "the supervisor serializes authorized decisions and restricted factual completion records. "
            "Never change limits, evaluator, labels, folds, harness, or unrelated recipe. "
            "Shell commands must start with rtk. Return only the requested structured report and new evidence. "
            "Narrative length and evidence/concern counts are concision targets, not validation limits. Aim comfortably below the targets; link detailed analysis. "
            "Preserve complete concerns and evidence rather than truncating them or changing decisions to fit a target. "
            "Do not restate standing constraints, unchanged rankings, scheduling, or historical conclusions. "
            f"{evidence_guidance}"
            "The supervisor owns scheduling, reservations, numerical verdicts, and mechanical transitions. "
            "For reviews, inspect the actual diff and critical source directly; concise reporting must not reduce substantive review. "
            "Use the supplied absolute python_executable for static checks; do not install dependencies or create an environment in the workspace. "
            "Workspaces are source copies, not Git branches. Compare with parent_source; never switch or commit the host checkout. "
            "No API fallback. Exit when the assignment is complete.\n\n"
            + json.dumps(context, indent=2)
        )
        atomic_write(directory / "stdin.txt", prompt)
        write_json(directory / "context-metrics.json", prompt_sizes(context, prompt))
        spec = read_json(directory / "spec.json")
        spec["command"] = command
        spec["provider"] = self.state["provider"]
        write_json(directory / "spec.json", spec)
        write_json(
            directory / "assignment.json", {"profile": profile, "context": context}
        )
        self.save()
        self.launch_worker(self.state["jobs"][job_id])

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
            else reference_snapshot(incumbent),
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
        if self.policy.require_source_change and not any(
            path.startswith("src/") for path in changed
        ):
            raise ValueError(
                "Candidate has no source change; scalar sweeps are outside this campaign"
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
        expected = read_json(
            Path(self.state["reference"]) / "fold_0/input_identity.json"
        )
        actual = read_json(directory / "input_identity.json")
        if actual != expected:
            raise ValueError(
                "Baseline data/preprocessing identity differs from historical control"
            )
        limits = self.state["limits"]
        if len(ledger["train"]) != limits["epochs"]:
            raise ValueError("Baseline did not replay the full training schedule")
        first = ledger["train"][0]
        steps = (
            math.ceil(
                math.ceil(
                    first["visits"] / self.state["config"]["trainer"]["batch_size"]
                )
                / self.state["config"]["trainer"]["accumulate_grad_batches"]
            )
            * limits["epochs"]
        )
        if steps != limits["optimizer_steps"]:
            raise ValueError(
                "Historical optimizer steps differ from the sealed schedule"
            )
        limits.update(
            training_triplets=ledger["training_triplets"],
            training_visits=ledger["training_visits"],
            training_cohort_sha256=first["cohort_sha256"],
            validation_cohort_sha256=ledger["validation"][0]["cohort_sha256"],
            validation_triplets=ledger["validation"][0]["triplets"],
        )
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
            self.state, exp, external_evidence(self.root, self.campaign)
        )

    def training(self, exp):
        blockers = launch_blockers(self.state, exp)
        if blockers:
            raise ValueError("Cannot launch: " + "; ".join(blockers))
        binding = self.review_binding(exp)
        approval = exp.get("implementation_review", {})
        if approval.get("decision") != "approve" or approval.get("binding") != binding:
            exp["stage"] = "awaiting_review"
            self.event(
                "review",
                {
                    "experiment": exp["id"],
                    "gate_error": None,
                    "refresh_reason": "Approval changed before launch; inspect the updated evidence",
                },
            )
            self.save()
            return
        source = Path(exp["source"])
        if manifest(source) != read_json(Path(exp["directory"]) / "manifest.json"):
            raise ValueError("Reviewed candidate source changed")
        if sha256(self.output / "limits.json") != self.state["sealed_sha256"]:
            raise ValueError("Sealed campaign limits changed")
        reference = self.state["incumbent"]
        if (
            sha256(Path(reference["run"]) / "fold_0/validation_predictions.csv")
            != reference["predictions_sha256"]
        ):
            raise ValueError("Incumbent predictions changed")
        config = dict(exp["config"])
        config["seed"] = binding["training_seed"]
        config["name"] = exp["id"]
        config["paths"] = {**config["paths"], "output_root": exp["directory"]}
        config_path = Path(exp["directory"]) / "config.yaml"
        atomic_write(config_path, yaml.safe_dump(config, sort_keys=False))
        exp.update(
            reference=dict(reference),
            seed=config["seed"],
            stage="launching",
            launch_binding=binding,
            active_idea_ids=binding["active_idea_ids"],
        )
        exp["reference_predictions_sha256"] = sha256(
            Path(reference["run"]) / "fold_0/validation_predictions.csv"
        )
        exp["comparison_keys"] = config_differences(
            read_json(Path(reference["run"]) / "effective_config.json"), config
        )
        validate_config(self.state["config"], config, self.policy)
        self.state["running"] = exp["id"]
        writable = [Path(exp["directory"])]
        cache = Path.home() / ".cache"
        if cache.exists():
            writable.append(cache)
        command = sandbox(
            [
                sys.executable,
                "-m",
                "scripts.pipeline.train",
                str(config_path),
                "--folds",
                "0",
                "--output-root",
                exp["directory"],
                "--run-id",
                "training",
            ],
            source,
            writable,
            gpu=True,
            readonly=[source, config_path],
        )
        self.submit(
            "training",
            source,
            command,
            payload={"experiment": exp["id"]},
            timeout=self.state["limits"]["max_run_seconds"]
            - exp["preparation_seconds"],
            gpu=self.state["gpu"],
            extra_env={"KNEE_AUTORESEARCH_LIMITS": str(self.output / "limits.json")},
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
            sha256(reference / "fold_0/validation_predictions.csv")
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
            "0",
            "--headers",
            self.state["headers"],
            "--bootstrap-draws",
            str(self.state["limits"]["bootstrap_draws"]),
            "--seed",
            str(self.state["limits"]["bootstrap_seed"]),
            "--allowed-config-differences",
            ",".join(exp["comparison_keys"]),
        ]
        identity_keys = comparison_identity_keys(self.state, exp, self.policy)
        if identity_keys:
            command += [
                "--allowed-input-identity-differences",
                ",".join(identity_keys),
            ]
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
                "comparison": str(directory / "comparison/comparison.json"),
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

    def completion(self, exp, decision, evidence):
        evidence = {**evidence, "acceptance_rule": dict(ACCEPTANCE_RULE)}
        exp.update(
            stage="awaiting_decision", eligible_decision=decision, evidence=evidence
        )
        write_json(
            Path(exp["directory"]) / "evaluation.json",
            {"eligible_decision": decision, **evidence},
        )
        self.event(
            "completion",
            {
                "experiment": exp["id"],
                "eligible_decision": decision,
                "evidence": evidence,
            },
        )

    def process_result(self, job, receipt):
        kind = job["kind"]
        directory = Path(job["path"])
        exp = self.state["experiments"].get(job["payload"].get("experiment"))
        failed = receipt["exit_code"] != 0 or receipt.get("error")
        if (
            kind == "comparison"
            and receipt.get("cancelled")
            and not (directory / "started.json").exists()
        ):
            exp["stage"] = "needs_comparison"
            return
        if receipt.get("cancelled") and kind not in {"training", "comparison"}:
            if kind == "completion_reporter":
                self.escalate_completion(
                    job,
                    "Reporter cancelled; orchestrator handles completion after resume",
                )
            elif kind == "developer":
                exp["stage"] = "needs_developer"
            elif kind == "config":
                exp["stage"] = "needs_config"
            elif kind == "researcher":
                self.state["last_refill_revision"] = -1
            return
        if kind == "training":
            if receipt["exit_code"] == 75:
                exp["stage"] = "prepared"
                self.state["running"] = None
                return
            self.dequeue_launched(exp)
            exp["training_receipt"] = receipt
            measured = {
                **receipt,
                "seconds": receipt["seconds"] + exp["preparation_seconds"]
                if receipt.get("seconds") is not None
                else None,
            }
            reasons = evaluate(Path(exp["directory"]), self.state["limits"], measured)
            if "training_seed" in exp.get("launch_binding", {}):
                try:
                    actual_seed = read_json(
                        Path(exp["directory"]) / "training/effective_config.json"
                    )["seed"]
                    if (
                        actual_seed != exp["launch_binding"]["training_seed"]
                        or actual_seed != exp["seed"]
                        or actual_seed != reference_seed(exp["reference"])
                    ):
                        reasons.append(
                            "Training seed differs from the recorded incumbent"
                        )
                except (OSError, KeyError, ValueError) as error:
                    reasons.append(
                        f"Missing or malformed training seed evidence: {error}"
                    )
            if not exp["launched"]:
                self.state["running"] = None
                exp["stage"] = "awaiting_review"
                self.event(
                    "review",
                    {
                        "experiment": exp["id"],
                        "gate_error": f"Launch failed: {reasons}",
                    },
                )
            elif reasons:
                self.completion(exp, "invalid", {"reasons": reasons})
            else:
                exp["stage"] = "needs_comparison"
            return
        if kind == "comparison":
            total_seconds = (
                exp["preparation_seconds"]
                + exp["training_receipt"]["seconds"]
                + (receipt.get("seconds") or 0)
            )
            exp["measured_compute_seconds"] = total_seconds
            failed = failed or total_seconds > self.state["limits"]["max_run_seconds"]
            if failed:
                self.completion(
                    exp,
                    "invalid",
                    {"reasons": [f"Comparison failed; inspect {directory}"]},
                )
            else:
                report = read_json(
                    Path(exp["directory"]) / "comparison/comparison.json"
                )
                diagnostics = read_diagnostics(
                    Path(exp["directory"]) / "comparison/comparison.json"
                )
                macro_bootstrap = (diagnostics or {}).get("macro_bootstrap")
                self.completion(
                    exp,
                    verdict(report, macro_bootstrap),
                    {
                        "comparison": str(
                            Path(exp["directory"]) / "comparison/comparison.json"
                        ),
                        "macro_auc": report["family_means"]["all"],
                        "adoption_bootstrap": macro_bootstrap,
                        "per_label": str(
                            diagnostics_path(
                                Path(exp["directory"]) / "comparison/comparison.json"
                            )
                        ),
                        "bootstrap": report["paired_report_group_bootstrap"][
                            "intervals"
                        ]["all"],
                    },
                )
            return
        if failed:
            if kind == "completion_reporter":
                self.escalate_completion(job, f"Reporter failed; inspect {directory}")
            if kind == "config":
                exp["stage"] = "awaiting_review"
                self.event(
                    "review",
                    {
                        "experiment": exp["id"],
                        "gate_error": f"Configuration failed; inspect {directory}",
                    },
                )
                return
            if kind == "developer":
                exp["stage"] = "needs_developer"
            self.state.update(
                status="paused",
                reason=f"{kind} failed; inspect {directory}. Fix the cause and use resume.",
            )
            return
        if kind == "calibration":
            self.seal()
        elif kind == "config":
            exp["preparation_seconds"] = receipt["seconds"]
            try:
                config = read_json(directory / "stdout.log")
                config.pop("config_file")
                config.pop("split_dir")
                validate_config(self.state["config"], config, self.policy)
                exp["config"] = config
                if not parent_changed(self.state, exp):
                    self.review_binding(exp)
                exp["stage"] = "awaiting_review"
                self.event("review", {"experiment": exp["id"], "gate_error": None})
            except (ValueError, KeyError) as error:
                exp["stage"] = "awaiting_review"
                self.event(
                    "review", {"experiment": exp["id"], "gate_error": str(error)}
                )
        else:
            report = providers.result(self.state["provider"], kind, directory)
            if kind == "researcher":
                self.state["last_refill_revision"] = self.state["evidence_revision"]
                if report["ideas"]:
                    self.event(
                        "proposals",
                        {"report": report, "artifact": str(directory / "report.json")},
                    )
            elif kind == "developer":
                exp["developer_report"] = report
                exp["integration"] = report["integration"]
                if report["status"] == "blocked":
                    exp["stage"] = "awaiting_review"
                    self.event(
                        "review",
                        {
                            "experiment": exp["id"],
                            "gate_error": report["blocker"] or "Developer blocked",
                        },
                    )
                elif self.state["status"] != "running":
                    exp["stage"] = "needs_prepare"
                else:
                    try:
                        self.prepare(exp, report)
                    except (OSError, ValueError, SyntaxError) as error:
                        exp["stage"] = "awaiting_review"
                        self.event(
                            "review",
                            {"experiment": exp["id"], "gate_error": str(error)},
                        )
            elif kind == "completion_reporter":
                self.apply_completion_report(job, report)
            else:
                self.apply_orchestrator(job["payload"]["event"], report)

    def completion_event(self, job):
        event_id = job["payload"]["event"]["id"]
        if not self.state["events"] or self.state["events"][0]["id"] != event_id:
            raise ValueError("Completion reporter event is stale")
        event = self.state["events"][0]
        if event["kind"] != "completion":
            raise ValueError("Completion reporter cannot handle another event type")
        routing = event.get("completion_reporting", {})
        if routing.get("job") != job["id"] or routing.get("status") != "assigned":
            raise ValueError("Completion reporter no longer owns the assigned event")
        return event

    def escalate_completion(self, job, reason):
        event = self.completion_event(job)
        record = {
            "status": "escalated",
            "job": job["id"],
            "reason": reason,
            "report": str(Path(job["path"]) / "report.json"),
        }
        event["completion_reporting"] = record
        self.state["experiments"][event["experiment"]]["completion_reporting"] = dict(
            record
        )

    def apply_completion_report(self, job, report):
        event = self.completion_event(job)
        reasons = completion_blockers(self.state, event)
        if report["status"] == "escalate" or report["concerns"] or reasons:
            reason = "; ".join([*reasons, *report["concerns"]]) or report["summary"]
            self.escalate_completion(job, reason)
            return
        exp = self.state["experiments"][event["experiment"]]
        self.apply_orchestrator(
            event,
            {
                "accepted_proposal_ids": [],
                "ranked_ids": None,
                "park_ids": [],
                "review": "none",
                "integration_review": None,
                "rationale": "Code recorded the fixed numerical result after restricted completion reporting. The incumbent and queue are unchanged; no research direction was closed.",
                "finding": report["summary"],
                "concerns": [],
                "evidence_refs": [
                    str(Path(job["path"]) / "report.json"),
                    *report["evidence_refs"],
                ],
            },
        )
        exp["completion_reporting"] = {
            "status": "reported",
            "job": job["id"],
            "report": str(Path(job["path"]) / "report.json"),
        }

    def apply_orchestrator(self, event, report):
        OrchestratorReport.model_validate(report)
        if not self.state["events"] or self.state["events"][0]["id"] != event["id"]:
            raise ValueError("Orchestrator event is stale")
        proposed = {
            idea["id"]: idea for idea in event.get("report", {}).get("ideas", [])
        }
        if len(proposed) != len(event.get("report", {}).get("ideas", [])):
            raise ValueError("Researcher supplied duplicate idea IDs")
        if proposed.keys() & self.state["ideas"].keys():
            raise ValueError("New research reports cannot overwrite existing ideas")
        available = {
            key: value
            for key, value in self.state["ideas"].items()
            if value["status"] == "proposed"
        } | proposed
        accepted = report["accepted_proposal_ids"]
        parked = report["park_ids"]
        if len(accepted) != len(set(accepted)) or not set(accepted) <= available.keys():
            raise ValueError("Invalid accepted proposal IDs")
        if not set(parked) <= set(self.state["queue"]):
            raise ValueError("Can only park queued ideas")
        expected = (set(self.state["queue"]) | set(accepted)) - set(parked)
        ranking = (
            self.state["queue"]
            if report["ranked_ids"] is None
            else report["ranked_ids"]
        )
        if (
            set(ranking) != expected
            or len(ranking) != len(expected)
            or len(ranking) > self.state["refill_to"]
        ):
            raise ValueError(
                "Ranking must contain each eligible live queue entry once, within the target"
            )
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            claimed
            and claimed["idea"] in parked
            and not (
                event["kind"] == "review" and event.get("experiment") == claimed["id"]
            )
        ):
            raise ValueError("An unrelated event cannot park work in progress")
        if (
            claimed
            and claimed["idea"] not in parked
            and (not ranking or ranking[0] != claimed["idea"])
        ):
            raise ValueError("Reserved queue head cannot be displaced")
        for idea_id in accepted:
            idea = available[idea_id]
            if not idea["novelty"].strip() or not idea["precedents"]:
                raise ValueError(
                    "A proposal needs precedent references and a material difference"
                )
            blockers = dependency_blockers(
                idea, self.state["ideas"], active_incumbent_ideas(self.state)
            )
            if blockers:
                raise ValueError(
                    f"Cannot queue {idea_id}: dependencies must be accepted and active in the incumbent: "
                    f"{'; '.join(blockers)}. Omit this ID from accepted_proposal_ids and ranked_ids; "
                    "it will remain proposed. Queue position does not resolve dependencies."
                )
        exp = self.state["experiments"].get(event.get("experiment"))
        binding = None
        if event["kind"] == "review":
            if report["review"] not in {"approve", "repair", "block"}:
                raise ValueError("Candidate review needs approve, repair, or block")
            if report["review"] == "approve" and (
                event.get("gate_error") or not exp.get("config")
            ):
                raise ValueError("Orchestrator cannot override a failed gate")
            if report["review"] == "repair" and exp["repair_count"] >= 1:
                raise ValueError("Only one bounded implementation repair is allowed")
            if report["review"] in {"repair", "approve"} and exp["idea"] in parked:
                raise ValueError("A repaired or approved head must remain reserved")
            if report["review"] == "block" and exp["idea"] not in parked:
                raise ValueError("Blocked review must explicitly park its idea")
            if report["review"] == "approve" or event.get("review_binding"):
                binding = self.review_binding(exp)
                if event.get("review_binding") != binding:
                    current_event = self.state["events"][0]
                    current_event.pop("review_binding", None)
                    current_event["refresh_reason"] = (
                        "Review inputs changed; inspect the updated evidence"
                    )
                    self.save()
                    return
        if event["kind"] == "completion" and (
            self.state["running"] != exp["id"]
            or exp["reference"] != self.state["incumbent"]
        ):
            raise ValueError(
                "Completion reference differs from the current incumbent; explicit recovery is required"
            )
        if event["kind"] != "review" and report["review"] != "none":
            raise ValueError(
                "Only candidate review events can approve, repair, or block"
            )
        for idea_id, idea in proposed.items():
            if idea_id not in self.state["ideas"]:
                self.state["ideas"][idea_id] = {
                    **idea,
                    "status": "proposed",
                    "created_revision": self.state["revision"],
                }
        for idea_id in accepted:
            self.state["ideas"][idea_id]["status"] = "queued"
        for idea_id in parked:
            self.state["ideas"][idea_id].update(
                status="blocked", blocked_reason=report["rationale"]
            )
            if claimed and claimed["idea"] == idea_id:
                claimed["stage"] = "blocked"
                self.state["claimed"] = None
            self.state["evidence_revision"] += 1
        self.state["queue"] = list(ranking)
        if report["ranked_ids"] is not None:
            for idea_id in ranking:
                self.state["ideas"][idea_id]["ranking_rationale"] = report["rationale"]
        if exp:
            exp.update(
                finding=report["finding"],
                concerns=report["concerns"],
                evidence_refs=report["evidence_refs"],
            )
        if event["kind"] == "review":
            exp["implementation_review"] = {
                "event": event["id"],
                "decision": report["review"],
                "concerns": report["concerns"],
                "integration_review": report["integration_review"],
                "binding": binding,
            }
            write_json(
                Path(exp["directory"]) / f"review-{event['id']}.json",
                exp["implementation_review"],
            )
            if report["review"] == "approve":
                exp["stage"] = "prepared"
                exp["active_idea_ids"] = binding["active_idea_ids"]
                self.state["ideas"][exp["idea"]]["status"] = "prepared"
            elif report["review"] == "repair":
                exp["stage"] = "repair_requested"
                exp["repair_instruction"] = report["rationale"]
        elif event["kind"] == "completion":
            exp.update(
                stage="completed",
                decision=event["eligible_decision"],
                rationale=report["rationale"],
            )
            self.state["ideas"][exp["idea"]].update(
                status="completed", decision=event["eligible_decision"]
            )
            self.state["running"] = None
            self.state["evidence_revision"] += 1
            if event["eligible_decision"] == "provisional_accept":
                run = Path(exp["directory"]) / "training"
                self.state["incumbent"] = {
                    "id": exp["id"],
                    "source": exp["source"],
                    "run": str(run),
                    "predictions_sha256": sha256(
                        run / "fold_0/validation_predictions.csv"
                    ),
                }
        self.state["findings"].append(
            {
                "event": event["id"],
                "finding": report["finding"],
                "rationale": report["rationale"],
                "concerns": report["concerns"],
                "evidence_refs": report["evidence_refs"],
            }
        )
        self.state["events"].pop(0)

    def harvest(self):
        for job in list(self.state["jobs"].values()):
            if job["processed"]:
                continue
            if job.get("application_error"):
                continue
            directory = Path(job["path"])
            receipt_path = directory / "receipt.json"
            if job["kind"] == "training":
                self.dequeue_launched(
                    self.state["experiments"][job["payload"]["experiment"]]
                )
            if receipt_path.exists():
                rollback = copy.deepcopy(self.state)
                job["processed"] = True
                try:
                    self.process_result(job, read_json(receipt_path))
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as error:
                    if job["kind"] in REPORTS:
                        if job["kind"] == "developer":
                            self.state["experiments"][job["payload"]["experiment"]][
                                "stage"
                            ] = "needs_developer"
                        if job["kind"] == "completion_reporter":
                            self.escalate_completion(
                                job, f"Reporter result processing failed: {error}"
                            )
                        elif job["kind"] == "orchestrator" and self.state["events"]:
                            self.state["events"][0]["previous_rejection"] = str(error)
                        self.state.update(
                            status="paused",
                            reason=f"Result processing failed for {job['id']}: {error}",
                        )
                        self.save()
                        continue
                    self.state = rollback
                    self.state["jobs"][job["id"]]["application_error"] = str(error)
                    self.state.update(
                        status="paused",
                        reason=f"Result processing failed for {job['id']}: {error}",
                    )
                    self.save()
                    return
                self.save()
                continue
            owner_path = directory / "owner.json"
            launcher_path = directory / "launcher.json"
            owner = read_json(owner_path) if owner_path.exists() else None
            launcher = read_json(launcher_path) if launcher_path.exists() else None
            if alive(owner) or alive(launcher):
                continue
            if time.time() - job["created_at"] < 10:
                continue
            if (directory / "started.json").exists():
                write_json(
                    receipt_path,
                    {
                        "exit_code": -1,
                        "error": "Worker lost; attempt will not be replayed",
                        "seconds": None,
                    },
                )
            elif self.state["status"] == "stopping":
                write_json(
                    receipt_path,
                    {
                        "exit_code": -1,
                        "cancelled": True,
                        "error": "Stopped before worker launch",
                        "seconds": 0,
                    },
                )
            else:
                self.launch_worker(job)

    def schedule(self):
        if self.state["status"] not in {"running", "calibrating"}:
            return
        if self.state["status"] == "calibrating":
            if not self.active("calibration"):
                base = Path(self.state["base"])
                destination = self.output / "baseline"
                self.submit(
                    "calibration",
                    base,
                    sandbox(
                        [
                            sys.executable,
                            "-m",
                            "knee_abnormality.autoresearch.cli",
                            "calibrate",
                            "--config",
                            str(base / "configs/training.yaml"),
                            "--destination",
                            str(destination),
                        ],
                        base,
                        [destination],
                        readonly=[base],
                    ),
                    timeout=14400,
                )
            return
        running = self.state["experiments"].get(self.state["running"])
        if running and running["stage"] == "needs_comparison":
            self.comparison(running)
        if self.active("orchestrator") or self.active("completion_reporter"):
            return
        if (
            self.state["running"]
            and self.state["experiments"][self.state["running"]]["stage"] == "launching"
        ):
            return
        if self.state["events"]:
            self.state["events"].sort(
                key=lambda event: {"completion": 0, "review": 1, "proposals": 2}[
                    event["kind"]
                ]
            )
            event = self.state["events"][0]
            if event["kind"] == "review" and not event.get("gate_error"):
                exp = self.state["experiments"][event["experiment"]]
                if parent_changed(self.state, exp) or not exp.get("integration"):
                    self.reconcile(exp)
                    self.state["events"].pop(0)
                    self.save()
                    return
                event["review_binding"] = self.review_binding(exp)
            if event["kind"] == "completion" and not event.get("completion_reporting"):
                blockers = completion_blockers(self.state, event)
                if not blockers and not event.get("previous_rejection"):
                    self.role(
                        "completion_reporter",
                        Path(self.state["base"]),
                        {"event": event},
                        payload={"event": event},
                    )
                    return
                event["completion_reporting"] = {
                    "status": "orchestrator",
                    "reasons": blockers or [event["previous_rejection"]],
                }
                self.state["experiments"][event["experiment"]][
                    "completion_reporting"
                ] = dict(event["completion_reporting"])
            context = {
                "event": event,
                "instructions": (
                    "Review evidence and interpret only this event. Use ranked_ids=null when the live queue order and membership stay unchanged. "
                    "When evidence requires a queue change, return a complete ranking of the LIVE queue plus accepted proposals minus explicitly parked IDs. "
                    "Keep any reserved head first. Target at most five. accepted_proposal_ids may contain IDs in this proposals event or existing proposed registry entries. "
                    "Consult proposal_admission: every dependency must be accepted and still present in active_incumbent_ideas. Historical acceptance of replaced behavior is insufficient. "
                    "Leave unresolved proposals out of accepted_proposal_ids and ranked_ids; they remain proposed automatically. Ranking them last does not make them eligible. "
                    "Retain surplus proposals without queueing weak ideas. Use review only for review events; otherwise none. "
                    "For completion interpret the fixed eligible_decision; the supervisor records it directly. Do not recompute it or return a verdict. "
                    "For review inspect changes.diff and source, including validation independence from training seed, coverage/exploitation, "
                    "leakage, actual mask accounting, preprocessing, and any added preparation cost. Failed gates cannot be approved. "
                    "Review the integration plan: preserve unrelated active ideas, verify every declared removal, and explain the result in integration_review for approval. "
                    "One concrete repair is allowed; block requires adding the idea to park_ids. "
                    "Do not park a developing/prepared idea for unrelated reranking. Empty arrays are valid where inapplicable. "
                    "Return new findings, unresolved concerns and evidence references. No scheduling instructions, repeated standing conditions, or unchanged queue explanations."
                ),
            }
            if event.get("experiment"):
                context["experiment"] = self.state["experiments"][event["experiment"]]
            self.role(
                "orchestrator",
                Path(self.state["base"]),
                context,
                payload={"event": event},
            )
            return
        running = self.state["experiments"].get(self.state["running"])
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            claimed
            and claimed["stage"]
            in {"needs_prepare", "needs_config", "repair_requested", "prepared"}
            and (
                parent_changed(self.state, claimed)
                or (claimed["stage"] == "prepared" and not claimed.get("integration"))
            )
        ):
            self.reconcile(claimed)
            return
        if claimed and claimed["stage"] == "needs_prepare":
            self.prepare(claimed, claimed["developer_report"])
            return
        if claimed and claimed["stage"] == "needs_config":
            self.config_gate(claimed)
            return
        if claimed and claimed["stage"] == "repair_requested":
            self.claim(repair=claimed)
            claimed = self.state["experiments"][self.state["claimed"]]
        if claimed and claimed["stage"] == "prepared":
            binding = self.review_binding(claimed)
            approval = claimed.get("implementation_review", {})
            if (
                approval.get("decision") != "approve"
                or approval.get("binding") != binding
            ):
                claimed["stage"] = "awaiting_review"
                self.event(
                    "review",
                    {
                        "experiment": claimed["id"],
                        "gate_error": None,
                        "refresh_reason": "Approval is absent or its parent, candidate or external evidence changed",
                    },
                )
                self.save()
                return
        if (
            claimed
            and claimed["stage"] == "prepared"
            and running is None
            and gpu_available(self.state["gpu"])
        ):
            self.training(claimed)
        if self.state["claimed"] is None and self.state["queue"]:
            self.claim()
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            claimed
            and claimed["stage"] == "needs_developer"
            and not self.active("developer")
        ):
            claimed["stage"] = "developing"
            self.role(
                "developer",
                Path(claimed["workspace"]),
                {
                    "idea": self.state["ideas"][claimed["idea"]],
                    "experiment": claimed,
                    "repair_instruction": self.state["experiments"][
                        claimed["repair_of"]
                    ].get("repair_instruction")
                    if claimed["repair_of"]
                    else None,
                    "completion": self.policy.developer_instructions,
                },
                payload={"experiment": claimed["id"]},
                repair=bool(claimed["repair_count"]),
            )
        if (
            len(self.state["queue"]) < self.state["refill_below"]
            and not self.active("researcher")
            and self.state["last_refill_revision"] != self.state["evidence_revision"]
        ):
            self.state["last_refill_revision"] = self.state["evidence_revision"]
            self.role(
                "researcher",
                Path(self.state["incumbent"]["source"]),
                {
                    "needed": self.state["refill_to"] - len(self.state["queue"]),
                    "instructions": (
                        "Use the compact mechanism registry, then retrieve relevant prior work and original reports through history_index before proposing. "
                        f"{self.policy.researcher_instructions} "
                        "Reference historic/general-backlog ownership and identify overlaps with ongoing work. Use unique stable lowercase-hyphen IDs. "
                        "Fewer ideas is fine; this refill will not immediately retrigger itself. Dependencies must name accepted ideas still active in the incumbent. Historical or replaced methods belong in precedents. "
                        "State what the proposal retains, adds and replaces, including how to handle an incumbent change during development. "
                        "Investigate primary sources as useful. Keep each proposal concise and put supporting evidence in referenced artifacts."
                    ),
                },
            )

    def run(self):
        with lock(self.output / "supervisor.lock"):
            write_json(self.output / "supervisor.json", process_identity(os.getpid()))
            self.project()

            def stop(signum, frame):
                write_json(self.output / "stop.json", {"abort": False})
                self.wakeup.set()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while True:
                try:
                    resume = self.output / "resume.json"
                    if resume.exists():
                        (self.output / "stop.json").unlink(missing_ok=True)
                        self.state.update(
                            status="running"
                            if self.state.get("sealed_sha256")
                            else "calibrating",
                            reason=None,
                        )
                        self.state["last_refill_revision"] = -1
                        for job in self.state["jobs"].values():
                            job.pop("application_error", None)
                        self.save()
                        resume.unlink()
                    request = self.output / "stop.json"
                    if request.exists() and self.state["status"] != "stopping":
                        self.state["status"] = "stopping"
                        self.save()
                    self.harvest()
                    if request.exists():
                        abort = read_json(request)["abort"]
                        if self.state["status"] != "stopping":
                            self.state["status"] = "stopping"
                            self.save()
                        for job in self.outstanding():
                            if abort or job["kind"] not in {"training", "comparison"}:
                                owner = Path(job["path"]) / "owner.json"
                                if owner.exists():
                                    terminate(read_json(owner))
                        if not self.outstanding():
                            self.state["status"] = "stopped"
                            self.save()
                            return
                    else:
                        self.schedule()
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as error:
                    self.state.update(
                        status="paused", reason=f"{type(error).__name__}: {error}"
                    )
                    self.save()
                self.wakeup.wait(5)
                self.wakeup.clear()
