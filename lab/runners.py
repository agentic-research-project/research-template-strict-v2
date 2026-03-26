"""
Runner abstractions for experiment execution.

  LocalRunner          — 로컬 subprocess 실행 (현재 동작 유지)
  GitHubActionsRunner  — GitHub Actions 워크플로우 트리거 + 결과 수집

공통 결과 형태 (RunResult):
  {
    "status":       "success | failed | smoke_failed | timeout | metrics_parse_error",
    "metrics":      {...},
    "stdout_lines": [...],
    "stderr_tail":  [...],
    "returncode":   int,
    "metadata": {
      "runner":       "local | github",
      "job_id":       "...",
      "duration_s":   float,
      "artifact_uri": "...",
      "job_url":      "...",
      "git_sha":      "..."
    }
  }

METRICS stdout 계약 (불변):
  train.py가 반드시 아래 형식으로 stdout에 출력해야 한다:
    METRICS:{...valid json...}
"""

import json
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path


# ──────────────────────────────────────────────────────────
# 추상 기반 클래스
# ──────────────────────────────────────────────────────────

class BaseRunner(ABC):
    """실험 실행 추상 인터페이스."""

    def is_ready(self) -> tuple[bool, str]:
        """runner가 실제 실험을 실행할 준비가 됐는지 반환한다.

        Returns:
            (True, "ok")                         — 실행 가능
            (False, "reason why not ready")      — 실행 불가

        side-effect 없이 호출해야 하며, 네트워크/파일 접근 없이 즉시 반환해야 한다.
        """
        return True, "ok"

    @abstractmethod
    def run_smoke(self, pkg_dir: Path) -> dict:
        """smoke test 실행. RunResult 반환."""
        ...

    @abstractmethod
    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        """학습 실행. RunResult 반환."""
        ...

    # ── 공통 유틸 ──────────────────────────────────────────

    @staticmethod
    def _parse_metrics(stdout_lines: list[str]) -> tuple[dict, bool]:
        """METRICS:{...} 라인을 파싱한다. (metrics_dict, parse_ok) 반환."""
        for line in stdout_lines:
            if re.match(r"^METRICS:\{.*\}$", line.strip()):
                try:
                    metrics = json.loads(line.strip()[len("METRICS:"):])
                    return metrics, True
                except Exception:
                    return {}, False
        return {}, False

    @staticmethod
    def _make_result(
        status: str,
        metrics: dict,
        stdout_lines: list[str],
        stderr_tail: list[str],
        returncode: int,
        metadata: dict,
    ) -> dict:
        return {
            "status":       status,
            "metrics":      metrics,
            "stdout_lines": stdout_lines,
            "stderr_tail":  stderr_tail,
            "returncode":   returncode,
            "metadata":     metadata,
        }

    @staticmethod
    def _make_failed_result(runner: str, reason: str) -> dict:
        return BaseRunner._make_result(
            status="failed", metrics={},
            stdout_lines=[], stderr_tail=[reason],
            returncode=-1,
            metadata=BaseRunner._empty_metadata(runner),
        )

    @staticmethod
    def _empty_metadata(runner: str) -> dict:
        return {
            "runner":       runner,
            "job_id":       "",
            "duration_s":   0.0,
            "artifact_uri": "",
            "job_url":      "",
            "git_sha":      "",
            "git_branch":   "",
        }


# ──────────────────────────────────────────────────────────
# LocalRunner — subprocess 기반 실행
# ──────────────────────────────────────────────────────────

class LocalRunner(BaseRunner):
    """로컬 subprocess 기반 실험 실행."""

    def run_smoke(self, pkg_dir: Path) -> dict:
        cmd = [sys.executable, "scripts/smoke_test.py", "--config", "configs/fast.yaml"]
        print(f"    [smoke test] {' '.join(cmd)}")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, cwd=str(pkg_dir)
            )
            duration = round(time.monotonic() - start, 2)
            ok     = proc.returncode == 0
            status = "success" if ok else "smoke_failed"
            print(f"    [smoke] {'PASS ✅' if ok else 'FAIL ❌'}")
            if not ok:
                print("    stderr:", proc.stderr[-500:])
            return self._make_result(
                status=status, metrics={},
                stdout_lines=proc.stdout.splitlines(),
                stderr_tail=proc.stderr.splitlines()[-100:],
                returncode=proc.returncode,
                metadata={**self._empty_metadata("local"), "duration_s": duration},
            )
        except subprocess.TimeoutExpired:
            return self._make_result(
                status="smoke_failed", metrics={},
                stdout_lines=[], stderr_tail=["smoke_test timeout"],
                returncode=-1,
                metadata=self._empty_metadata("local"),
            )
        except Exception as e:
            print(f"    [smoke] 예외: {e}")
            return self._make_result(
                status="smoke_failed", metrics={},
                stdout_lines=[], stderr_tail=[str(e)],
                returncode=-1,
                metadata=self._empty_metadata("local"),
            )

    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        cmd = [sys.executable, "train.py", "--config", config_file]
        print(f"\n    [실험 실행] cd {pkg_dir} && {' '.join(cmd)}")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(pkg_dir)
            )
            duration     = round(time.monotonic() - start, 2)
            stdout_lines = proc.stdout.splitlines()
            stderr_lines = proc.stderr.splitlines()

            metrics, parse_ok = self._parse_metrics(stdout_lines)
            if proc.returncode != 0:
                status = "failed"
            elif not parse_ok:
                status = "metrics_parse_error"
            else:
                status = "success"

            return self._make_result(
                status=status, metrics=metrics,
                stdout_lines=stdout_lines, stderr_tail=stderr_lines[-200:],
                returncode=proc.returncode,
                metadata={**self._empty_metadata("local"), "duration_s": duration},
            )
        except subprocess.TimeoutExpired:
            return self._make_result(
                status="timeout", metrics={},
                stdout_lines=[], stderr_tail=[],
                returncode=-1,
                metadata={**self._empty_metadata("local"), "duration_s": float(timeout)},
            )
        except Exception as e:
            return self._make_result(
                status="failed", metrics={},
                stdout_lines=[], stderr_tail=[str(e)],
                returncode=-1,
                metadata=self._empty_metadata("local"),
            )


# ──────────────────────────────────────────────────────────
# GitHubActionsRunner — GitHub Actions 기반 실행 (skeleton)
# ──────────────────────────────────────────────────────────

class GitHubActionsRunner(BaseRunner):
    """
    GitHub Actions 워크플로우 트리거 + 결과 수집.

    필요 환경 변수 (또는 생성자 인자):
      GITHUB_TOKEN    personal access token (workflow + contents scope)
      GITHUB_OWNER    레포지토리 소유자 (예: myorg)
      GITHUB_REPO     레포지토리 이름 (예: my-research)
      GITHUB_REF      트리거 브랜치 (기본값: main)
      GITHUB_WORKFLOW 워크플로우 파일명 (기본값: experiment.yml)

    동작 흐름:
      1. 실험 패키지 코드를 git commit/push (CI가 checkout할 수 있도록)
      2. workflow_dispatch 트리거
      3. run이 나타날 때까지 polling
      4. run 완료 대기 (timeout)
      5. artifact 다운로드 → results/vN/ 에 저장
      6. RunResult 반환
    """

    _API = "https://api.github.com"

    def __init__(
        self,
        token: str = "",
        owner: str = "",
        repo: str = "",
        ref: str = "main",
        workflow: str = "experiment.yml",
        poll_interval: int = 30,
        max_poll_secs: int = 10800,
    ):
        self.token         = token
        self.owner         = owner
        self.repo          = repo
        self.ref           = ref
        self.workflow      = workflow
        self.poll_interval = poll_interval
        self.max_poll_secs = max_poll_secs

    # ─────────────────────────────────────────────────────
    # is_ready
    # ─────────────────────────────────────────────────────
    def is_ready(self) -> tuple[bool, str]:
        if not self.token:
            return False, "GITHUB_TOKEN 미설정"
        if not self.owner or not self.repo:
            return False, "GITHUB_OWNER / GITHUB_REPO 미설정"
        return True, "ok"

    # ─────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return {
            "Authorization":        f"Bearer {self.token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _base(self) -> str:
        return f"{self._API}/repos/{self.owner}/{self.repo}"

    def _push_pkg(self, pkg_dir: Path) -> str:
        """실험 패키지를 git commit/push하고 HEAD SHA를 반환한다."""
        result = subprocess.run(
            ["git", "add", str(pkg_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git add 실패: {result.stderr}")

        # 변경사항이 있을 때만 commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True,
        ).stdout.strip()

        if status:
            subprocess.run(
                ["git", "commit", "-m",
                 f"chore: push experiment package {pkg_dir.name} for CI"],
                capture_output=True, text=True, check=True,
            )
            subprocess.run(
                ["git", "push", "origin", self.ref],
                capture_output=True, text=True, check=True,
            )
            print(f"  [GitHubRunner] 패키지 push 완료: {pkg_dir}")

        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha

    def _dispatch(self, pkg_dir: Path, config_file: str, smoke_only: bool = False) -> None:
        import urllib.request
        payload = json.dumps({
            "ref": self.ref,
            "inputs": {
                "experiment_pkg": str(pkg_dir),
                "config_file":    config_file,
                "smoke_only":     "true" if smoke_only else "false",
            },
        }).encode()
        req = urllib.request.Request(
            f"{self._base()}/actions/workflows/{self.workflow}/dispatches",
            data=payload,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            if resp.status != 204:
                raise RuntimeError(f"workflow_dispatch 실패: {resp.status}")

    def _find_run(self, after_ts: str) -> int | None:
        """after_ts 이후에 생성된 workflow run ID를 반환한다."""
        import urllib.request
        url = (f"{self._base()}/actions/runs"
               f"?event=workflow_dispatch&branch={self.ref}&per_page=5")
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            runs = json.loads(resp.read()).get("workflow_runs", [])
        for run in runs:
            if run.get("created_at", "") >= after_ts:
                return run["id"]
        return None

    def _poll_run(self, run_id: int, timeout: int) -> dict:
        """run이 완료될 때까지 polling. 완료된 run dict 반환."""
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            req = urllib.request.Request(
                f"{self._base()}/actions/runs/{run_id}",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req) as resp:
                run = json.loads(resp.read())
            if run.get("status") == "completed":
                return run
            print(f"  [GitHubRunner] run {run_id}: {run.get('status')} ...")
        raise TimeoutError(f"run {run_id} timeout ({timeout}s)")

    def _download_artifacts(self, run_id: int, pkg_dir: Path) -> dict:
        """실험 결과 artifact를 다운로드하고 results/vN/ 에 저장한다."""
        import io
        import urllib.request
        import zipfile

        # experiments/{slug}/results/vN/
        results_dir = pkg_dir.parent.parent / "results" / pkg_dir.name
        results_dir.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(
            f"{self._base()}/actions/runs/{run_id}/artifacts",
            headers=self._headers(),
        )
        with urllib.request.urlopen(req) as resp:
            artifacts = json.loads(resp.read()).get("artifacts", [])

        result_summary = {}
        runner_metadata = {}

        for art in artifacts:
            if art["name"] not in ("experiment-results", "train-artifacts"):
                continue
            dl_url = f"{self._base()}/actions/artifacts/{art['id']}/zip"
            req = urllib.request.Request(dl_url, headers=self._headers())
            with urllib.request.urlopen(req) as resp:
                zdata = resp.read()

            with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                for name in zf.namelist():
                    data = zf.read(name)
                    fname = Path(name).name
                    if fname == "result_summary.json":
                        result_summary = json.loads(data)
                        (results_dir / "result_summary.json").write_bytes(data)
                    elif fname == "runner_metadata.json":
                        runner_metadata = json.loads(data)
                        (results_dir / "runner_metadata.json").write_bytes(data)
                    elif name.startswith("artifacts/"):
                        target = pkg_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)

        return result_summary, runner_metadata

    def _build_result_from_run(
        self,
        run: dict,
        result_summary: dict,
        runner_metadata: dict,
    ) -> dict:
        conclusion = run.get("conclusion", "failure")
        status = "success" if conclusion == "success" else "failed"

        metrics: dict = {}
        pm = result_summary.get("primary_metric", {})
        if pm.get("name"):
            metrics[pm["name"]] = pm.get("value", 0.0)
        for sm in result_summary.get("secondary_metrics", []):
            metrics[sm["name"]] = sm.get("value", 0.0)

        meta = {
            **self._empty_metadata("github"),
            "job_id":       str(run.get("id", "")),
            "duration_s":   runner_metadata.get("duration_s", 0.0),
            "artifact_uri": run.get("artifacts_url", ""),
            "job_url":      run.get("html_url", ""),
            "git_sha":      run.get("head_sha", ""),
            "git_branch":   self.ref,
        }
        stdout_lines = [f"METRICS:{json.dumps(metrics)}"] if metrics else []

        return self._make_result(
            status=status,
            metrics=metrics,
            stdout_lines=stdout_lines,
            stderr_tail=[],
            returncode=0 if conclusion == "success" else 1,
            metadata=meta,
        )

    # ─────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────
    def run_smoke(self, pkg_dir: Path) -> dict:
        """실험 패키지를 push → smoke-only 워크플로우 트리거 → 결과 반환."""
        import datetime
        try:
            self._push_pkg(pkg_dir)
            after_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            self._dispatch(pkg_dir, "configs/fast.yaml", smoke_only=True)
            print(f"  [GitHubRunner] smoke workflow 트리거 완료")

            # run 찾기 (최대 60초)
            run_id = None
            deadline = time.time() + 60
            while time.time() < deadline:
                time.sleep(5)
                run_id = self._find_run(after_ts)
                if run_id:
                    break
            if not run_id:
                return self._make_failed_result("github", "smoke run을 찾을 수 없음")

            run = self._poll_run(run_id, timeout=300)  # smoke: 5분
            conclusion = run.get("conclusion", "failure")
            status = "success" if conclusion == "success" else "smoke_failed"
            meta = {
                "runner":   "github",
                "job_id":   str(run_id),
                "job_url":  run.get("html_url", ""),
                "git_sha":  run.get("head_sha", ""),
            }
            return self._make_result(
                status=status, metrics={},
                stdout_lines=[], stderr_tail=[],
                returncode=0 if conclusion == "success" else 1,
                metadata={**self._empty_metadata("github"), **meta},
            )
        except Exception as e:
            return self._make_failed_result("github", str(e))

    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        """실험 패키지를 push → 전체 학습 워크플로우 트리거 → 결과 수집."""
        import datetime
        try:
            self._push_pkg(pkg_dir)
            after_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            self._dispatch(pkg_dir, config_file, smoke_only=False)
            print(f"  [GitHubRunner] train workflow 트리거 완료")

            run_id = None
            deadline = time.time() + 90
            while time.time() < deadline:
                time.sleep(10)
                run_id = self._find_run(after_ts)
                if run_id:
                    break
            if not run_id:
                return self._make_failed_result("github", "train run을 찾을 수 없음")

            print(f"  [GitHubRunner] run_id={run_id} 폴링 중 (timeout={timeout}s)...")
            run = self._poll_run(run_id, timeout=timeout)
            result_summary, runner_metadata = self._download_artifacts(run_id, pkg_dir)
            return self._build_result_from_run(run, result_summary, runner_metadata)

        except TimeoutError:
            return self._make_failed_result("github", "timeout")
        except Exception as e:
            return self._make_failed_result("github", str(e))

    @classmethod
    def from_config(cls, cfg: dict) -> "GitHubActionsRunner":
        return cls(
            token         = cfg.get("github_token", ""),
            owner         = cfg.get("github_owner", ""),
            repo          = cfg.get("github_repo", ""),
            ref           = cfg.get("github_ref", "main"),
            workflow      = cfg.get("github_workflow", "experiment.yml"),
            poll_interval = cfg.get("github_poll_interval", 30),
            max_poll_secs = cfg.get("github_max_poll_secs", 10800),
        )


# ──────────────────────────────────────────────────────────
# 팩토리
# ──────────────────────────────────────────────────────────

def create_runner(runner_type: str = "local", runner_config: dict | None = None) -> BaseRunner:
    """runner_type에 따라 Runner 인스턴스를 생성한다.

    Args:
        runner_type:   "local" | "github"
        runner_config: GitHubActionsRunner에 필요한 설정 dict
                       {"github_token": ..., "github_owner": ..., "github_repo": ...}
    """
    runner_config = runner_config or {}
    if runner_type == "local":
        return LocalRunner()
    elif runner_type == "github":
        return GitHubActionsRunner.from_config(runner_config)
    else:
        raise ValueError(f"알 수 없는 runner_type: {runner_type!r}  (local | github)")
