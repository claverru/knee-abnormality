import copy
import subprocess
import sys
import time
from pathlib import Path

from knee_abnormality.autoresearch.campaign.binding import (
    external_evidence,
    reference_seed,
)
from knee_abnormality.autoresearch.campaign.reports import (
    REPORTS,
)
from knee_abnormality.autoresearch.campaign.rules import ACCEPTANCE_RULE, EVALUATION
from knee_abnormality.autoresearch.runtime import providers
from knee_abnormality.autoresearch.runtime.isolation import (
    environment,
)
from knee_abnormality.autoresearch.runtime.layout import campaign_fold
from knee_abnormality.autoresearch.runtime.storage import (
    alive,
    atomic_write,
    optional_json,
    read_json,
    spawn,
    write_json,
)
from knee_abnormality.autoresearch.supervisor.common import (
    REJECTION_LIMIT,
    RETRY_DELAYS,
    identifier,
    work_item,
)
from knee_abnormality.autoresearch.supervisor.context import build_context, prompt_sizes
from knee_abnormality.autoresearch.supervisor.prompts import role_prompt


class JobMixin:
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
        spawn(
            ["knee_abnormality.autoresearch.cli", "job", "--job-dir", str(directory)],
            self.root,
            directory / "worker.log",
            directory / "launcher.json",
            environment(self.root),
        )

    def submit(self, *args, **kwargs):
        job_id, directory = self.create_job(*args, **kwargs)
        self.save()
        self.launch_worker(self.state["jobs"][job_id])
        return job_id, directory

    def role(self, role, source: Path, context, *, payload=None):
        if role != "completion_reporter":
            context = {
                **context,
                "external_evidence": external_evidence(
                    self.root, self.campaign, campaign_fold(self.state)
                ),
            }
        job_id, directory = self.create_job(role, source, [], payload=payload)
        if role == "completion_reporter":
            self.route_completion(
                payload["event"], {"status": "assigned", "job": job_id}
            )
        profile = providers.resolved(
            self.state["provider"],
            self.state["profiles"][self.state["provider"]][role],
        )
        command = providers.invocation(
            self.state["provider"],
            REPORTS[role],
            profile,
            source,
            directory,
            self.policy.editable_directories if role == "developer" else None,
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
        context["shared_evaluation"] = {**EVALUATION, "fold": campaign_fold(self.state)}
        if role != "completion_reporter":
            context["campaign_policy"] = self.policy.model_dump(
                mode="json",
                exclude={"developer_instructions", "researcher_instructions"},
            )
        guidance = (Path(self.state["base"]) / "AGENTS.md").read_text()
        prompt = role_prompt(guidance, brief, contract, context)
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

    def waiting(self):
        return time.time() < self.state.get("retry_after", 0)

    def provider_wait(self, job, problem):
        if problem["kind"] == "transient":
            retries = self.state.setdefault("provider_retries", {})
            key = work_item(job)
            retries[key] = retries.get(key, 0) + 1
            if retries[key] > len(RETRY_DELAYS):
                return False
            until = time.time() + RETRY_DELAYS[retries[key] - 1]
            attempt = f"retry {retries[key]}/{len(RETRY_DELAYS)}"
        elif problem["kind"] == "limit":
            until = problem["retry_after"]
            attempt = "retry after the reset"
        else:
            return False
        self.state["retry_after"] = max(until, self.state.get("retry_after", 0))
        wait = time.strftime(
            "%Y-%m-%d %H:%M:%S %Z", time.localtime(self.state["retry_after"])
        )
        self.state["reason"] = (
            f"{job['kind']} {problem['kind']} provider error: {problem['message'][:500]}. "
            f"Agent jobs wait until {wait} ({attempt})."
        )
        return True

    def reject(self, job, error):
        job["processed"] = True
        self.state.get("provider_retries", {}).pop(work_item(job), None)
        if job["kind"] == "developer":
            target = self.state["experiments"][job["payload"]["experiment"]]
            target["stage"] = "needs_developer"
        elif job["kind"] == "researcher":
            target = self.state["research"]
            self.state["last_refill_revision"] = -1
        else:
            target = next(
                (
                    event
                    for event in self.state["events"]
                    if event["id"] == job["payload"]["event"]["id"]
                ),
                None,
            )
            if target is None:
                return
        target["previous_rejection"] = str(error)
        target["rejections"] = target.get("rejections", 0) + 1
        if target["rejections"] >= REJECTION_LIMIT:
            self.pause(
                f"Report rejected {target['rejections']} consecutive times for {job['id']}: {error}",
                "rejection",
            )

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
                    if job["kind"] in {"orchestrator", "developer", "researcher"}:
                        self.state = rollback
                        self.reject(self.state["jobs"][job["id"]], error)
                        self.save()
                        return
                    if job["kind"] == "completion_reporter":
                        self.escalate_completion(
                            job, f"Reporter result processing failed: {error}"
                        )
                        self.save()
                        continue
                    if job["kind"] in REPORTS:
                        self.pause(f"Result processing failed for {job['id']}: {error}")
                        self.save()
                        continue
                    self.state = rollback
                    self.state["jobs"][job["id"]]["application_error"] = str(error)
                    self.pause(f"Result processing failed for {job['id']}: {error}")
                    self.save()
                    return
                self.save()
                continue
            owner_path = directory / "owner.json"
            launcher_path = directory / "launcher.json"
            if alive(optional_json(owner_path)) or alive(optional_json(launcher_path)):
                if job["kind"] == "training":
                    self.watch_training(job)
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
