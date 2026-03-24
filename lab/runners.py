"""
Runner abstractions for experiment execution.

  LocalRunner   — 로컬 subprocess 실행 (현재 동작 유지)
  GitLabRunner  — GitLab CI/CD 트리거 + 결과 수집 (skeleton, TODO 경계 명시)

공통 결과 형태 (RunResult):
  {
    "status":       "success | failed | smoke_failed | timeout | metrics_parse_error",
    "metrics":      {...},
    "stdout_lines": [...],
    "stderr_tail":  [...],
    "returncode":   int,
    "metadata": {
      "runner":       "local | gitlab",
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
    def _empty_metadata(runner: str) -> dict:
        return {
            "runner":       runner,
            "job_id":       "",
            "duration_s":   0.0,
            "artifact_uri": "",
            "job_url":      "",
            "git_sha":      "",
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
# GitLabRunner — GitLab CI/CD 기반 실행 (skeleton)
# ──────────────────────────────────────────────────────────

class GitLabRunner(BaseRunner):
    """
    GitLab CI/CD 파이프라인 트리거 + 결과 수집.

    현재 상태: interface + metadata shape 완성, 실제 API 호출은 TODO.

    필요 환경 변수 (또는 생성자 인자):
      GITLAB_URL         GitLab 인스턴스 URL (예: https://gitlab.example.com)
      GITLAB_TOKEN       personal access token (api scope)
      GITLAB_PROJECT_ID  프로젝트 ID (정수 또는 URL-encoded namespace/path)
      GITLAB_REF         트리거할 브랜치/태그 (기본값: main)
    """

    def __init__(
        self,
        gitlab_url: str = "",
        token: str = "",
        project_id: str = "",
        ref: str = "main",
        poll_interval: int = 30,
        max_poll_secs: int = 10800,
    ):
        self.gitlab_url    = gitlab_url
        self.token         = token
        self.project_id    = project_id
        self.ref           = ref
        self.poll_interval = poll_interval
        self.max_poll_secs = max_poll_secs

    # ── 구현 완료 전까지 False 반환 ─────────────────────────
    _FULLY_IMPLEMENTED = False   # run_smoke / run_train 구현 완료 시 True로 변경

    def is_ready(self) -> tuple[bool, str]:
        """GitLabRunner는 구현 완료 전까지 실행 불가 상태를 반환한다.

        side-effect 없이 즉시 반환한다 (네트워크/파일 접근 없음).
        _FULLY_IMPLEMENTED = True로 변경하면 준비됨으로 전환된다.
        """
        if not self._FULLY_IMPLEMENTED:
            return False, (
                "GitLabRunner 미구현 (skeleton only). "
                "lab/runners.py의 run_smoke / run_train을 구현 후 "
                "_FULLY_IMPLEMENTED = True 로 변경하세요."
            )
        return True, "ok"

    def run_smoke(self, pkg_dir: Path) -> dict:
        """
        TODO: GitLab smoke 실행
          1. POST /projects/{id}/trigger/pipeline
               variables: {"SMOKE": "true", "PKG_DIR": str(pkg_dir)}
          2. poll GET /projects/{id}/pipelines/{pipeline_id}
               until status in ("success", "failed", "canceled")
          3. GET job logs → parse stdout for smoke result
          4. return RunResult with metadata.runner="gitlab"

        현재는 NotImplementedError 발생.
        """
        raise NotImplementedError(
            "GitLabRunner.run_smoke() — 구현 필요:\n"
            "  1. POST /projects/{id}/trigger/pipeline (SMOKE=true)\n"
            "  2. poll pipeline status\n"
            "  3. collect job logs → returncode 파싱\n"
            "  4. return RunResult"
        )

    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        """
        TODO: GitLab full train 실행
          1. POST /projects/{id}/trigger/pipeline
               variables: {"PKG_DIR": str(pkg_dir), "CONFIG": config_file}
          2. poll GET /projects/{id}/pipelines/{pipeline_id}
               until status in ("success", "failed", "canceled") or timeout
          3. GET /projects/{id}/jobs/{train_job_id}/artifacts → result_summary.json
          4. GET /projects/{id}/jobs/{train_job_id}/trace → METRICS 라인 파싱
          5. return RunResult with metadata:
               runner="gitlab", job_id, job_url, artifact_uri, git_sha, duration_s

        현재는 NotImplementedError 발생.
        """
        raise NotImplementedError(
            "GitLabRunner.run_train() — 구현 필요:\n"
            "  1. POST /projects/{id}/trigger/pipeline\n"
            "  2. poll pipeline status (timeout 고려)\n"
            "  3. artifact 수집 → result_summary.json\n"
            "  4. job trace → METRICS 파싱\n"
            "  5. return RunResult with full metadata"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "GitLabRunner":
        """config dict에서 GitLabRunner를 생성한다."""
        return cls(
            gitlab_url    = cfg.get("gitlab_url", ""),
            token         = cfg.get("gitlab_token", ""),
            project_id    = cfg.get("gitlab_project_id", ""),
            ref           = cfg.get("gitlab_ref", "main"),
            poll_interval = cfg.get("gitlab_poll_interval", 30),
            max_poll_secs = cfg.get("gitlab_max_poll_secs", 10800),
        )


# ──────────────────────────────────────────────────────────
# 팩토리
# ──────────────────────────────────────────────────────────

def create_runner(runner_type: str = "local", runner_config: dict | None = None) -> BaseRunner:
    """runner_type에 따라 Runner 인스턴스를 생성한다.

    Args:
        runner_type:   "local" | "gitlab"
        runner_config: GitLabRunner에 필요한 설정 dict
                       {"gitlab_url": ..., "gitlab_token": ..., "gitlab_project_id": ...}
    """
    runner_config = runner_config or {}
    if runner_type == "local":
        return LocalRunner()
    elif runner_type == "gitlab":
        return GitLabRunner.from_config(runner_config)
    else:
        raise ValueError(f"알 수 없는 runner_type: {runner_type!r}  (local | gitlab)")
