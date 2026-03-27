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
      "runner":         "local | github",
      "job_id":         str,        # GitHub: run_id, local: ""
      "pipeline_id":    str,        # GitHub: run_number, local: ""
      "dispatch_id":    str,        # GitHubActionsRunner가 생성한 UUID (local: "")
      "runner_name":    str,        # GitHub self-hosted runner 이름 (GitHub only)
      "duration_s":     float,
      "artifact_uri":   str,        # GitHub: artifacts API URL
      "job_url":        str,        # GitHub: html_url (브라우저로 바로 열 수 있는 URL)
      "git_sha":        str,
      "git_branch":     str,
      "experiment_pkg": str,
      "started_at":     str,        # ISO8601
      "finished_at":    str,        # ISO8601
    }
  }

METRICS stdout 계약 (불변):
  train.py가 반드시 아래 형식으로 stdout에 출력해야 한다:
    METRICS:{...valid json...}

artifact 계약 (GitHubActionsRunner):
  runner는 experiment-results artifact에서 아래 두 파일을 수집한다:
    artifacts/metrics/final_metrics.json  — raw metric dict
    runner_metadata.json                  — 실행 메타데이터
  canonical result_summary.json는 research_loop.py가 생성한다.

GitHub 실행 전제조건:
  환경변수 (또는 CLI 인자):
    GITHUB_TOKEN    — personal access token (workflow + contents scope 필요)
    GITHUB_OWNER    — 레포지토리 소유자 (예: myorg)
    GITHUB_REPO     — 레포지토리 이름 (예: my-research)
    GITHUB_REF      — 트리거 브랜치 (기본값: main)
    GITHUB_WORKFLOW — 워크플로우 파일명 (기본값: experiment.yml)
  GitHub Secrets (workflow 내에서 필요):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
  인프라:
    self-hosted GPU runner (runs-on: [self-hosted, gpu]) 필요
"""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from lab.config import result_version_dir, slug_from_pkg, version_from_pkg


# ──────────────────────────────────────────────────────────
# 추상 기반 클래스
# ──────────────────────────────────────────────────────────

class BaseRunner(ABC):
    """실험 실행 추상 인터페이스.

    RunResult 계약:
      status        : "success" | "failed" | "smoke_failed" | "timeout" | "metrics_parse_error"
      metrics       : {"metric_name": float, ...}  — METRICS:{} stdout에서 파싱
      stdout_lines  : 전체 stdout 줄 목록 (최대 수천 줄)
      stderr_tail   : stderr 마지막 200줄 (디버깅용)
      returncode    : 프로세스 exit code (GitHub: 0=success, 1=failure, -1=runner 오류)
      metadata      : runner_metadata dict (공통 스키마)
    """

    def is_ready(self) -> tuple[bool, str]:
        """runner가 실험을 실행할 준비가 됐는지 반환.
        side-effect 없이 즉시 반환해야 한다.
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
            "metadata":     BaseRunner._sanitize_metadata(metadata),
        }

    @staticmethod
    def _make_failed_result(runner: str, reason: str, extra_meta: dict | None = None) -> dict:
        """실패 RunResult를 생성한다. reason은 사람이 읽고 바로 원인 파악 가능해야 한다."""
        meta = BaseRunner._empty_metadata(runner)
        if extra_meta:
            meta.update(extra_meta)
        return BaseRunner._make_result(
            status="failed", metrics={},
            stdout_lines=[], stderr_tail=[reason],
            returncode=-1,
            metadata=meta,
        )

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """metadata에서 토큰/API 키 등 민감 정보를 제거한다.

        URL에 포함된 토큰, 환경변수 값 등이 result_summary.json에
        기록되지 않도록 sanitize한다.
        """
        import re
        sanitized = {}
        # 토큰 패턴: ghp_, gho_, github_pat_, sk-, AIza 등
        _SECRET_PATTERN = re.compile(
            r"(ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|"
            r"github_pat_[A-Za-z0-9_]{82}|"
            r"sk-[A-Za-z0-9]{48,}|"
            r"AIza[A-Za-z0-9\-_]{35})",
        )
        for k, v in meta.items():
            if isinstance(v, str):
                sanitized[k] = _SECRET_PATTERN.sub("***REDACTED***", v)
            elif isinstance(v, dict):
                sanitized[k] = BaseRunner._sanitize_metadata(v)
            else:
                sanitized[k] = v
        return sanitized

    @staticmethod
    def _empty_metadata(runner: str) -> dict:
        return {
            "runner":         runner,
            "job_id":         "",
            "pipeline_id":    "",
            "dispatch_id":    "",
            "duration_s":     0.0,
            "artifact_uri":   "",
            "job_url":        "",
            "git_sha":        "",
            "git_branch":     "",
            "experiment_pkg": "",
            "started_at":     "",
            "finished_at":    "",
        }


# ──────────────────────────────────────────────────────────
# LocalRunner — subprocess 기반 실행
# ──────────────────────────────────────────────────────────

class LocalRunner(BaseRunner):
    """로컬 subprocess 기반 실험 실행.

    smoke: scripts/smoke_test.py --config configs/fast.yaml
    train: train.py --config <config_file>
    """

    @staticmethod
    def _env_with_pythonpath(pkg_dir: Path) -> dict:
        """pkg_dir을 PYTHONPATH에 추가한 환경변수 딕셔너리."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(pkg_dir) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def run_smoke(self, pkg_dir: Path) -> dict:
        cmd = [sys.executable, "scripts/smoke_test.py", "--config", "configs/fast.yaml"]
        print(f"    [smoke test] {' '.join(cmd)}")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
                cwd=str(pkg_dir), env=self._env_with_pythonpath(pkg_dir)
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
                metadata={
                    **self._empty_metadata("local"),
                    "experiment_pkg": str(pkg_dir),
                    "duration_s":     duration,
                },
            )
        except subprocess.TimeoutExpired:
            return self._make_result(
                status="smoke_failed", metrics={},
                stdout_lines=[], stderr_tail=["smoke_test timeout (180s)"],
                returncode=-1,
                metadata={**self._empty_metadata("local"), "experiment_pkg": str(pkg_dir)},
            )
        except Exception as e:
            print(f"    [smoke] 예외: {e}")
            return self._make_result(
                status="smoke_failed", metrics={},
                stdout_lines=[], stderr_tail=[f"smoke 실행 예외: {e}"],
                returncode=-1,
                metadata={**self._empty_metadata("local"), "experiment_pkg": str(pkg_dir)},
            )

    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        import datetime as _dt
        cmd = [sys.executable, "train.py", "--config", config_file]
        print(f"\n    [실험 실행] cd {pkg_dir} && {' '.join(cmd)}")
        start     = time.monotonic()
        started_at = _dt.datetime.utcnow().isoformat()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(pkg_dir), env=self._env_with_pythonpath(pkg_dir)
            )
            duration      = round(time.monotonic() - start, 2)
            finished_at   = _dt.datetime.utcnow().isoformat()
            stdout_lines  = proc.stdout.splitlines()
            stderr_lines  = proc.stderr.splitlines()

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
                metadata={
                    **self._empty_metadata("local"),
                    "experiment_pkg": str(pkg_dir),
                    "duration_s":     duration,
                    "started_at":     started_at,
                    "finished_at":    finished_at,
                },
            )
        except subprocess.TimeoutExpired:
            return self._make_result(
                status="timeout", metrics={},
                stdout_lines=[], stderr_tail=[f"train timeout ({timeout}s)"],
                returncode=-1,
                metadata={
                    **self._empty_metadata("local"),
                    "experiment_pkg": str(pkg_dir),
                    "duration_s":     float(timeout),
                    "started_at":     started_at,
                },
            )
        except Exception as e:
            return self._make_result(
                status="failed", metrics={},
                stdout_lines=[], stderr_tail=[f"train 실행 예외: {e}"],
                returncode=-1,
                metadata={**self._empty_metadata("local"), "experiment_pkg": str(pkg_dir)},
            )


# ──────────────────────────────────────────────────────────
# GitHubActionsRunner — GitHub Actions 기반 실행
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
      1. dispatch_id 생성 (uuid, 이 run의 고유 식별자)
      2. 실험 패키지 코드를 git commit/push (CI checkout용)
      3. workflow_dispatch 트리거 (dispatch_id 포함)
      4. run이 나타날 때까지 polling (SHA + after_ts 기반 식별)
         복수 후보 발견 시 명시적 실패 (조용한 첫 번째 선택 금지)
      5. run 완료 대기 (timeout)
      6. experiment-results artifact 다운로드
         → artifacts/metrics/final_metrics.json
         → runner_metadata.json
      7. RunResult 반환 (canonical summary는 research_loop.py가 생성)

    주의:
      - canonical result_summary.json은 이 runner가 생성하지 않는다.
      - research_loop.py의 _build_result_summary()가 RunResult + experiment_spec.json 기반으로 생성한다.
    """

    _API = "https://api.github.com"

    @staticmethod
    def _detect_from_git_remote() -> dict:
        """git remote URL에서 owner, repo, token을 자동 감지한다.
        예: https://TOKEN@github.com/OWNER/REPO.git → {token, owner, repo}
        """
        import re as _re
        try:
            result = subprocess.run(
                ["git", "remote", "-v"], capture_output=True, text=True,
            )
            for line in result.stdout.strip().split("\n"):
                # https://TOKEN@github.com/OWNER/REPO.git (push)
                m = _re.search(
                    r"https://([^@]+)@github\.com/([^/]+)/([^/\s]+?)(?:\.git)?\s",
                    line,
                )
                if m:
                    return {"token": m.group(1), "owner": m.group(2), "repo": m.group(3)}
                # https://github.com/OWNER/REPO.git (token 없는 경우)
                m = _re.search(
                    r"https://github\.com/([^/]+)/([^/\s]+?)(?:\.git)?\s",
                    line,
                )
                if m:
                    return {"owner": m.group(1), "repo": m.group(2)}
        except Exception:
            pass
        return {}

    def __init__(
        self,
        token: str = "",
        owner: str = "",
        repo: str = "",
        ref: str = "main",
        workflow: str = "experiment.yml",
        poll_interval: int = 30,
        max_poll_secs: int = 10800,
        project_dir: str = "",
    ):
        # 명시적 값 → 환경변수 → git remote 자동 감지 순서
        detected = {}
        if not (token and owner and repo):
            detected = self._detect_from_git_remote()

        self.token         = token or os.environ.get("GITHUB_TOKEN", "") or detected.get("token", "")
        self.owner         = owner or os.environ.get("GITHUB_OWNER", "") or detected.get("owner", "")
        self.repo          = repo  or os.environ.get("GITHUB_REPO", "")  or detected.get("repo", "")
        self.ref           = ref
        self.workflow      = workflow
        self.poll_interval = poll_interval
        self.max_poll_secs = max_poll_secs
        self.project_dir   = project_dir or str(Path(__file__).resolve().parent.parent)

    # ─────────────────────────────────────────────────────
    # is_ready
    # ─────────────────────────────────────────────────────
    def is_ready(self) -> tuple[bool, str]:
        missing = []
        if not self.token:
            missing.append("GITHUB_TOKEN")
        if not self.owner:
            missing.append("GITHUB_OWNER")
        if not self.repo:
            missing.append("GITHUB_REPO")
        if missing:
            return False, f"미설정 환경변수: {', '.join(missing)}"
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

    def _run_url(self, run_id: int | str) -> str:
        """브라우저로 열 수 있는 GitHub Actions run URL."""
        return f"https://github.com/{self.owner}/{self.repo}/actions/runs/{run_id}"

    def _push_pkg(self, pkg_dir: Path) -> str:
        """실험 패키지를 git commit/push하고 HEAD SHA를 반환한다."""
        result = subprocess.run(
            ["git", "add", str(pkg_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git add 실패: {result.stderr.strip()}")

        changed = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True,
        ).stdout.strip()

        if changed:
            commit = subprocess.run(
                ["git", "commit", "-m",
                 f"chore: push experiment package {pkg_dir.name} for CI"],
                capture_output=True, text=True, check=True,
            )
            push = subprocess.run(
                ["git", "push", "origin", self.ref],
                capture_output=True, text=True,
            )
            if push.returncode != 0:
                raise RuntimeError(f"git push 실패: {push.stderr.strip()}")
            print(f"  [GitHubRunner] 패키지 push 완료: {pkg_dir}")
        else:
            print(f"  [GitHubRunner] 변경사항 없음 — 기존 commit 사용")

        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha

    def _dispatch(
        self,
        pkg_dir: Path,
        config_file: str,
        smoke_only: bool = False,
        dispatch_id: str = "",
    ) -> None:
        """workflow_dispatch를 트리거한다. dispatch_id를 workflow input으로 전달."""
        import urllib.request
        payload = json.dumps({
            "ref": self.ref,
            "inputs": {
                "experiment_pkg": str(pkg_dir),
                "config_file":    config_file,
                "smoke_only":     "true" if smoke_only else "false",
                "dispatch_id":    dispatch_id,
                "project_dir":    self.project_dir,
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
                raise RuntimeError(f"workflow_dispatch 실패: HTTP {resp.status}")

    def _check_run_dispatch_id(self, run_id: int, dispatch_id: str) -> bool | None:
        """개별 run의 setup job 로그에서 dispatch_id를 검증한다.

        workflow의 'Print run info' 단계에서 dispatch_id를 출력하므로,
        로그에서 해당 dispatch_id가 존재하는지 확인한다.

        반환값:
          True  — dispatch_id가 로그에서 확인됨
          False — 로그에 접근했으나 dispatch_id가 없음 (다른 run)
          None  — 로그에 아직 접근할 수 없음 (run이 시작 전이거나 로그 미생성)
        """
        import urllib.request
        try:
            url = f"{self._base()}/actions/runs/{run_id}/logs"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req) as resp:
                import io, zipfile
                zdata = resp.read()
                with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                    for name in zf.namelist():
                        content = zf.read(name).decode("utf-8", errors="replace")
                        if f"dispatch_id: {dispatch_id}" in content:
                            return True
            # 로그에 접근했으나 dispatch_id를 찾지 못함 → 확실히 불일치
            return False
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                # 로그가 아직 생성되지 않음 (run 시작 전)
                return None
            # 403 등 구조적 접근 불가
            print(f"    [경고] run {run_id} 로그 접근 실패: HTTP {e.code}")
            return None
        except Exception as e:
            # zip 파싱 실패 등 — 로그 미생성일 수 있음
            print(f"    [경고] run {run_id} 로그 검증 중 예외: {e}")
            return None

    def _find_run(self, after_ts: str, expected_sha: str, dispatch_id: str) -> int:
        """dispatch_id + SHA + after_ts 기반으로 정확한 run을 식별한다.

        식별 전략:
        1. after_ts 이후 생성된 run을 시간 기반 1차 필터
        2. SHA 기반 2차 필터
        3. 모든 후보(1개 포함)에 대해 dispatch_id 로그 검증 수행
        4. dispatch_id 검증을 통과한 run이 정확히 1개여야 확정
        5. 검증 통과 0개 → 로그 미생성일 수 있으므로 재시도
        6. 검증 통과 2개 이상 → 명시적 실패
        7. 120초 내 미발견 시 RuntimeError
        """
        import urllib.request
        deadline = time.time() + 120  # 2분 대기
        print(f"  [GitHubRunner] run 탐색 중 dispatch_id={dispatch_id} sha={expected_sha[:8]}")
        while time.time() < deadline:
            time.sleep(10)
            url = (f"{self._base()}/actions/runs"
                   f"?event=workflow_dispatch&branch={self.ref}&per_page=10")
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req) as resp:
                runs = json.loads(resp.read()).get("workflow_runs", [])

            # after_ts 기반 1차 필터
            candidates = [r for r in runs if r.get("created_at", "") >= after_ts]

            # sha 기반 2차 필터
            sha_matched = [r for r in candidates if r.get("head_sha") == expected_sha]
            if sha_matched:
                candidates = sha_matched

            if len(candidates) == 0:
                continue  # 아직 run이 생성되지 않음

            # 모든 후보에 대해 dispatch_id 로그 검증 (1개여도 반드시 검증)
            print(f"  [GitHubRunner] {len(candidates)}개 후보 발견, dispatch_id 로그 검증 시작")
            verified = []
            logs_pending = False
            for c in candidates:
                cid = c["id"]
                if c.get("status") in ("completed", "in_progress"):
                    check = self._check_run_dispatch_id(cid, dispatch_id)
                    if check is True:
                        verified.append(c)
                        print(f"    run_id={cid} → dispatch_id 일치 ✓")
                    elif check is False:
                        print(f"    run_id={cid} → dispatch_id 불일치 ✗")
                    else:
                        # None: 로그 미생성 → 재시도 대상
                        logs_pending = True
                        print(f"    run_id={cid} → 로그 미접근 (재시도)")
                else:
                    # queued 등 아직 시작 전 → 로그 검증 불가, 재시도
                    logs_pending = True
                    print(f"    run_id={cid} → status={c.get('status')} (대기 중)")

            if len(verified) == 1:
                run_id = verified[0]["id"]
                print(f"  [GitHubRunner] dispatch_id 검증으로 run 확정: run_id={run_id} "
                      f"url={self._run_url(run_id)}")
                return run_id
            elif len(verified) > 1:
                ids = [r["id"] for r in verified]
                raise RuntimeError(
                    f"run 식별 실패: dispatch_id={dispatch_id}에 대해 "
                    f"{len(verified)}개 검증 통과 {ids} — 명시적 실패"
                )

            # verified == 0: 로그 미생성이면 재시도, 모두 불일치면 즉시 실패
            if not logs_pending:
                raise RuntimeError(
                    f"run 식별 실패: {len(candidates)}개 후보 모두 dispatch_id={dispatch_id} "
                    f"검증 불일치 — 해당 dispatch에 대응하는 run 없음"
                )
            # logs_pending=True → 아직 로그가 준비되지 않은 run 있음, 재시도
            continue

        raise RuntimeError(
            f"run 식별 timeout: dispatch_id={dispatch_id} "
            f"(120s 내 dispatch_id 검증 통과 run 미발견, branch={self.ref})"
        )

    def _poll_run(self, run_id: int, timeout: int) -> dict:
        """run이 completed 될 때까지 polling. 완료된 run dict 반환."""
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

    def _download_artifacts(self, run_id: int, pkg_dir: Path) -> tuple[dict, dict]:
        """experiment-results artifact에서 final_metrics.json + runner_metadata.json을 복원.

        반환: (final_metrics, runner_metadata)
        파일 없으면 RuntimeError (조용한 fallback 없음).
        복원 위치: experiments/{slug}/results/{vN}/
        """
        import io, urllib.request, zipfile

        # topic-local results 저장 경로
        slug = slug_from_pkg(pkg_dir)
        ver = version_from_pkg(pkg_dir)
        res_ver_dir = result_version_dir(slug, ver)
        res_ver_dir.mkdir(parents=True, exist_ok=True)

        # artifact 목록 조회
        req = urllib.request.Request(
            f"{self._base()}/actions/runs/{run_id}/artifacts",
            headers=self._headers(),
        )
        with urllib.request.urlopen(req) as resp:
            artifacts = json.loads(resp.read()).get("artifacts", [])

        target = next((a for a in artifacts if a["name"] == "experiment-results"), None)
        if not target:
            raise RuntimeError(
                f"experiment-results artifact 없음 (run_id={run_id}, "
                f"url={self._run_url(run_id)})"
            )

        # zip 다운로드
        dl_url = f"{self._base()}/actions/artifacts/{target['id']}/zip"
        req = urllib.request.Request(dl_url, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            zdata = resp.read()

        final_metrics: dict   = {}
        runner_metadata: dict = {}

        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            for name in zf.namelist():
                data  = zf.read(name)
                fname = Path(name).name
                if fname == "final_metrics.json":
                    final_metrics = json.loads(data)
                    (res_ver_dir / "final_metrics.json").write_bytes(data)
                elif fname == "runner_metadata.json":
                    runner_metadata = json.loads(data)
                    (res_ver_dir / "runner_metadata.json").write_bytes(data)

        # 필수 파일 누락 시 명시적 실패
        if not final_metrics:
            raise RuntimeError(
                f"final_metrics.json artifact에 없음 (run_id={run_id}) — "
                f"train 단계에서 METRICS 파싱 실패했을 가능성 있음"
            )
        if not runner_metadata:
            raise RuntimeError(
                f"runner_metadata.json artifact에 없음 (run_id={run_id})"
            )

        print(f"  [GitHubRunner] artifact 복원 완료 → {res_ver_dir}")
        return final_metrics, runner_metadata

    def _build_result_from_run(
        self,
        run: dict,
        final_metrics: dict,
        runner_metadata: dict,
        dispatch_id: str = "",
    ) -> dict:
        """RunResult를 구성한다. canonical result_summary는 research_loop.py가 생성."""
        conclusion = run.get("conclusion", "failure")
        status     = "success" if conclusion == "success" else "failed"
        run_id_str = str(run.get("id", ""))
        run_url    = run.get("html_url", self._run_url(run_id_str))

        # artifact의 runner_metadata를 기반으로 시작하여 workflow가 생성한 필드를
        # 손실 없이 보존하고, API에서 얻은 최신/정확한 값으로 오버라이드한다.
        meta = {
            **self._empty_metadata("github"),
            **runner_metadata,
            "job_id":         run_id_str,
            "pipeline_id":    str(run.get("run_number", "")),
            "dispatch_id":    dispatch_id or runner_metadata.get("dispatch_id", ""),
            "artifact_uri":   run.get("artifacts_url", f"{self._base()}/actions/runs/{run_id_str}/artifacts"),
            "job_url":        run_url,
            "git_sha":        run.get("head_sha", "") or runner_metadata.get("git_sha", ""),
            "git_branch":     self.ref,
            "finished_at":    runner_metadata.get("finished_at", run.get("updated_at", "")),
        }
        # METRICS 계약 유지: stdout_lines에 METRICS 라인 포함 (research_loop._build_result_summary 파싱용)
        stdout_lines = [f"METRICS:{json.dumps(final_metrics)}"] if final_metrics else []

        return self._make_result(
            status=status,
            metrics=final_metrics,
            stdout_lines=stdout_lines,
            stderr_tail=[],
            returncode=0 if conclusion == "success" else 1,
            metadata=meta,
        )

    # ─────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────

    def run_smoke(self, pkg_dir: Path) -> dict:
        """push → smoke-only 워크플로우 트리거 → 결과 반환.

        실패 원인 구분:
          dispatch 실패     : git push 오류 또는 workflow_dispatch 오류
          run 식별 실패     : 120s 내 미발견 또는 복수 후보
          poll timeout      : smoke 5분 초과
          workflow failure  : conclusion != success
        """
        import datetime as _dt
        dispatch_id = f"smoke_{uuid.uuid4().hex[:12]}"
        print(f"  [GitHubRunner] smoke 시작 dispatch_id={dispatch_id} pkg={pkg_dir}")

        # 1. git push
        try:
            sha = self._push_pkg(pkg_dir)
        except Exception as e:
            return self._make_failed_result(
                "github", f"dispatch 실패 - git push 오류: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        after_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 2. workflow_dispatch
        try:
            self._dispatch(pkg_dir, "configs/fast.yaml", smoke_only=True, dispatch_id=dispatch_id)
            print(f"  [GitHubRunner] workflow dispatch 완료 (smoke)")
        except Exception as e:
            return self._make_failed_result(
                "github", f"dispatch 실패 - workflow_dispatch 오류: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        # 3. run 식별
        try:
            run_id = self._find_run(after_ts, sha, dispatch_id)
        except RuntimeError as e:
            return self._make_failed_result(
                "github", f"run 식별 실패: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        # 4. poll (smoke: 5분)
        try:
            run = self._poll_run(run_id, timeout=300)
        except TimeoutError:
            return self._make_result(
                status="timeout", metrics={},
                stdout_lines=[],
                stderr_tail=[f"smoke timeout 300s (dispatch_id={dispatch_id} run_id={run_id})"],
                returncode=-1,
                metadata={
                    **self._empty_metadata("github"),
                    "job_id":      str(run_id),
                    "dispatch_id": dispatch_id,
                    "job_url":     self._run_url(run_id),
                    "experiment_pkg": str(pkg_dir),
                },
            )

        conclusion = run.get("conclusion", "failure")
        status     = "success" if conclusion == "success" else "smoke_failed"
        run_url    = run.get("html_url", self._run_url(run_id))
        return self._make_result(
            status=status, metrics={},
            stdout_lines=[], stderr_tail=[],
            returncode=0 if conclusion == "success" else 1,
            metadata={
                **self._empty_metadata("github"),
                "job_id":         str(run_id),
                "pipeline_id":    str(run.get("run_number", "")),
                "dispatch_id":    dispatch_id,
                "job_url":        run_url,
                "artifact_uri":   run.get("artifacts_url", ""),
                "git_sha":        run.get("head_sha", sha),
                "git_branch":     self.ref,
                "experiment_pkg": str(pkg_dir),
                "started_at":     run.get("created_at", ""),
                "finished_at":    run.get("updated_at", ""),
            },
        )

    def run_train(
        self,
        pkg_dir: Path,
        config_file: str = "configs/default.yaml",
        timeout: int = 7200,
    ) -> dict:
        """push → 전체 학습 워크플로우 트리거 → artifact 수집 → RunResult 반환.

        실패 원인 구분:
          dispatch 실패          : git push 오류 또는 workflow_dispatch 오류
          run 식별 실패          : 120s 내 미발견 또는 복수 후보
          poll timeout           : train timeout 초과
          workflow failure       : conclusion != success
          artifact 다운로드 실패 : experiment-results artifact 없음
          metrics parse 실패     : final_metrics.json 없음
        """
        import datetime as _dt
        dispatch_id = f"train_{uuid.uuid4().hex[:12]}"
        print(f"  [GitHubRunner] train 시작 dispatch_id={dispatch_id} pkg={pkg_dir}")

        # 1. git push
        try:
            sha = self._push_pkg(pkg_dir)
        except Exception as e:
            return self._make_failed_result(
                "github", f"dispatch 실패 - git push 오류: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        after_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 2. workflow_dispatch
        try:
            self._dispatch(pkg_dir, config_file, smoke_only=False, dispatch_id=dispatch_id)
            print(f"  [GitHubRunner] workflow dispatch 완료 (train)")
        except Exception as e:
            return self._make_failed_result(
                "github", f"dispatch 실패 - workflow_dispatch 오류: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        # 3. run 식별
        try:
            run_id = self._find_run(after_ts, sha, dispatch_id)
        except RuntimeError as e:
            return self._make_failed_result(
                "github", f"run 식별 실패: {e}",
                {"dispatch_id": dispatch_id, "experiment_pkg": str(pkg_dir)},
            )

        print(f"  [GitHubRunner] run_id={run_id} 폴링 중 (timeout={timeout}s) dispatch_id={dispatch_id}")

        # 4. poll
        try:
            run = self._poll_run(run_id, timeout=timeout)
        except TimeoutError:
            return self._make_result(
                status="timeout", metrics={},
                stdout_lines=[],
                stderr_tail=[f"train timeout {timeout}s (dispatch_id={dispatch_id} run_id={run_id} url={self._run_url(run_id)})"],
                returncode=-1,
                metadata={
                    **self._empty_metadata("github"),
                    "job_id":      str(run_id),
                    "dispatch_id": dispatch_id,
                    "job_url":     self._run_url(run_id),
                    "experiment_pkg": str(pkg_dir),
                },
            )

        # 5. workflow 실패 처리
        if run.get("conclusion") != "success":
            run_url = run.get("html_url", self._run_url(run_id))
            return self._make_result(
                status="failed", metrics={},
                stdout_lines=[],
                stderr_tail=[
                    f"workflow 실패: conclusion={run.get('conclusion')} "
                    f"dispatch_id={dispatch_id} url={run_url}"
                ],
                returncode=1,
                metadata={
                    **self._empty_metadata("github"),
                    "job_id":      str(run_id),
                    "dispatch_id": dispatch_id,
                    "job_url":     run_url,
                    "git_sha":     run.get("head_sha", sha),
                    "experiment_pkg": str(pkg_dir),
                },
            )

        # 6. artifact 다운로드
        try:
            final_metrics, runner_metadata = self._download_artifacts(run_id, pkg_dir)
        except RuntimeError as e:
            return self._make_failed_result(
                "github", f"artifact 다운로드 실패: {e}",
                {"dispatch_id": dispatch_id, "job_id": str(run_id),
                 "job_url": run.get("html_url", self._run_url(run_id)),
                 "experiment_pkg": str(pkg_dir)},
            )
        except Exception as e:
            return self._make_failed_result(
                "github", f"artifact 처리 중 예외: {e}",
                {"dispatch_id": dispatch_id, "job_id": str(run_id),
                 "experiment_pkg": str(pkg_dir)},
            )

        return self._build_result_from_run(run, final_metrics, runner_metadata, dispatch_id)

    @classmethod
    def from_config(cls, cfg: dict) -> "GitHubActionsRunner":
        """cfg dict → 환경변수 → git remote 자동 감지 순서로 값을 결정한다."""
        return cls(
            token         = cfg.get("github_token")   or os.environ.get("GITHUB_TOKEN", ""),
            owner         = cfg.get("github_owner")   or os.environ.get("GITHUB_OWNER", ""),
            repo          = cfg.get("github_repo")    or os.environ.get("GITHUB_REPO", ""),
            ref           = cfg.get("github_ref")     or os.environ.get("GITHUB_REF", "main"),
            workflow      = cfg.get("github_workflow") or os.environ.get("GITHUB_WORKFLOW", "experiment.yml"),
            poll_interval = cfg.get("github_poll_interval", 30),
            max_poll_secs = cfg.get("github_max_poll_secs", 10800),
            project_dir   = cfg.get("project_dir", ""),
        )


# ──────────────────────────────────────────────────────────
# 팩토리
# ──────────────────────────────────────────────────────────

def create_runner(runner_type: str = "github", runner_config: dict | None = None) -> BaseRunner:
    """runner_type에 따라 Runner 인스턴스를 생성한다.

    Args:
        runner_type:   "local" | "github"
        runner_config: GitHubActionsRunner에 필요한 설정 dict
                       {
                         "github_token":    str,  # 또는 GITHUB_TOKEN env
                         "github_owner":    str,  # 또는 GITHUB_OWNER env
                         "github_repo":     str,  # 또는 GITHUB_REPO env
                         "github_ref":      str,  # 기본값: "main"
                         "github_workflow": str,  # 기본값: "experiment.yml"
                       }
    """
    runner_config = runner_config or {}
    if runner_type == "local":
        return LocalRunner()
    elif runner_type == "github":
        return GitHubActionsRunner.from_config(runner_config)
    else:
        raise ValueError(f"알 수 없는 runner_type: {runner_type!r}  (local | github)")
