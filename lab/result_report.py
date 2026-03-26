"""
결과 보고서 PDF 생성

research_loop 완료 후 실험 결과를 A4 PDF로 정리한다.
- PDF: experiments/{slug}/results/result_report.pdf

섹션:
  1. 연구 주제 + 가설 요약
  2. 실험 결과 (primary metric, 달성 여부)
  3. 버전별 이력 테이블
  4. 분석 요약 (GPT/Gemini/Claude 판정)
  5. 최종 결정 + 후속 조치
"""

import json
from datetime import datetime
from pathlib import Path

from lab.config import results_dir, slug_from_pkg


# ──────────────────────────────────────────────────────────
# 색상 상수
# ──────────────────────────────────────────────────────────

C_BG       = "#1A1A2E"
C_HEADER   = "#0F3460"
C_ACCENT   = "#E94560"
C_SUCCESS  = "#27AE60"
C_WARN     = "#E67E22"
C_FAIL     = "#C0392B"
C_CARD     = "#2C3E50"
C_LIGHT    = "#F0F4F8"
C_GRAY     = "#95A5A6"
C_WHITE    = "white"
C_BLACK    = "#222222"


# ──────────────────────────────────────────────────────────
# 폰트 설정 (user_approval.py와 동일 패턴)
# ──────────────────────────────────────────────────────────

def _setup_fonts():
    import glob, sys
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    patterns = [
        f"{sys.prefix}/lib/python*/site-packages/koreanize_matplotlib/fonts/NanumBarunGothic.ttf",
        f"{sys.prefix}/lib/python*/site-packages/koreanize_matplotlib/fonts/*.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            continue
        regular = next((m for m in matches if "Bold" not in m and "bold" not in m), matches[0])
        bold = next((m for m in matches if "Bold" in m or "bold" in m), regular)
        try:
            pdfmetrics.registerFont(TTFont("KR", regular))
            pdfmetrics.registerFont(TTFont("KR-Bold", bold))
            return "KR", "KR-Bold"
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


# ──────────────────────────────────────────────────────────
# 드로잉 헬퍼
# ──────────────────────────────────────────────────────────

def _color(c):
    from reportlab.lib.colors import HexColor, white
    if c == "white":
        return white
    return HexColor(c)


def _rect(cv, x, y, w, h, fill, radius=0):
    cv.setFillColor(_color(fill))
    cv.setStrokeColor(_color(fill))
    if radius:
        cv.roundRect(x, y, w, h, radius, fill=1, stroke=0)
    else:
        cv.rect(x, y, w, h, fill=1, stroke=0)


def _text(cv, text, x, y, font, size, color):
    cv.setFont(font, size)
    cv.setFillColor(_color(color))
    cv.drawString(x, y, str(text))
    return y


def _wrap_text(cv, text, x, y, max_w, font, size, color, line_h=None):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    if line_h is None:
        line_h = size * 1.4
    words = str(text or "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if stringWidth(test, font, size) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    cv.setFont(font, size)
    cv.setFillColor(_color(color))
    cy = y
    for line in lines:
        cv.drawString(x, cy, line)
        cy -= line_h
    return cy


# ──────────────────────────────────────────────────────────
# 결과 보고서 PDF 생성
# ──────────────────────────────────────────────────────────

def generate_result_pdf(
    final_summary: dict,
    run_history: list[dict],
    topic: dict,
    hypothesis: dict,
    decision: dict | None,
    output_path: Path,
) -> Path:
    """실험 결과를 A4 PDF로 생성한다.

    Args:
        final_summary: 최종 result_summary dict
        run_history:   각 round의 run_entry 목록
        topic:         topic_analysis.json
        hypothesis:    hypothesis.json
        decision:      최종 decision dict (Claude 결정)
        output_path:   PDF 저장 경로
    Returns:
        생성된 PDF 파일 경로
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    FN, FB = _setup_fonts()
    W, H = A4
    ML = 10 * mm
    CW = W - 2 * ML

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv = rl_canvas.Canvas(str(output_path), pagesize=A4)

    # ── 데이터 추출 ──────────────────────────────────────
    inp = topic.get("input", {})
    hyp = hypothesis.get("hypothesis", {})
    primary = final_summary.get("primary_metric", {})
    met = primary.get("met", False)
    status = final_summary.get("status", "unknown")
    path = decision.get("path", "?") if decision else "?"

    # ── 1. 헤더 ──────────────────────────────────────────
    _rect(cv, 0, H - 50, W, 50, C_BG)
    _text(cv, "Research Experiment Report", ML + 5, H - 32, FB, 16, C_WHITE)
    _text(cv, datetime.now().strftime("%Y-%m-%d %H:%M"), W - ML - 100, H - 32, FN, 9, C_GRAY)

    y = H - 70

    # ── 2. 연구 주제 ────────────────────────────────────
    _rect(cv, ML, y - 55, CW, 55, C_LIGHT, radius=4)
    _text(cv, "Research Topic", ML + 8, y - 15, FB, 11, C_HEADER)
    y_topic = _wrap_text(cv, inp.get("topic", "N/A"), ML + 8, y - 30, CW - 16, FN, 9, C_BLACK)
    y_topic = _wrap_text(cv, f"Details: {inp.get('details', '')}", ML + 8, y_topic - 2, CW - 16, FN, 8, C_GRAY)
    y = y - 65

    # ── 3. 가설 요약 ────────────────────────────────────
    _rect(cv, ML, y - 60, CW, 60, C_LIGHT, radius=4)
    _text(cv, "Hypothesis", ML + 8, y - 15, FB, 11, C_HEADER)
    stmt = hyp.get("statement_kr", hyp.get("statement", "N/A"))
    _wrap_text(cv, stmt, ML + 8, y - 30, CW - 16, FN, 9, C_BLACK)
    mechanism = hyp.get("expected_mechanism", hyp.get("key_mechanism", hyp.get("mechanism", "")))
    if mechanism:
        _wrap_text(cv, f"Mechanism: {mechanism}", ML + 8, y - 48, CW - 16, FN, 8, C_GRAY)
    y = y - 70

    # ── 4. 최종 결과 (큰 박스) ──────────────────────────
    result_color = C_SUCCESS if met else C_WARN if status == "success" else C_FAIL
    _rect(cv, ML, y - 75, CW, 75, result_color, radius=6)

    result_label = "TARGET ACHIEVED" if met else f"Path {path}" if path != "?" else status.upper()
    _text(cv, result_label, ML + 12, y - 22, FB, 14, C_WHITE)

    _text(cv, f"{primary.get('name', '?')}:", ML + 12, y - 42, FN, 11, C_WHITE)
    _text(cv, f"{primary.get('value', 0):.4f}", ML + 80, y - 42, FB, 14, C_WHITE)
    _text(cv, f"target: {primary.get('target', 0):.2f}", ML + 180, y - 42, FN, 10, C_WHITE)

    # secondary metrics
    sec = final_summary.get("secondary_metrics", [])
    if sec:
        sec_text = " | ".join(f"{m['name']}: {m['value']:.4f}" for m in sec[:4])
        _text(cv, sec_text, ML + 12, y - 60, FN, 9, C_WHITE)

    y = y - 85

    # ── 5. 버전별 이력 테이블 ────────────────────────────
    _text(cv, "Run History", ML, y, FB, 11, C_HEADER)
    y -= 18

    # 테이블 헤더
    col_x = [ML, ML + 40, ML + 90, ML + 200, ML + 280, ML + 360]
    col_labels = ["Ver", "Status", "Primary Metric", "Decision", "Consensus", "GPT Path"]
    _rect(cv, ML, y - 14, CW, 16, C_CARD)
    for i, label in enumerate(col_labels):
        _text(cv, label, col_x[i] + 3, y - 10, FB, 7.5, C_WHITE)
    y -= 16

    # 테이블 행
    for idx, entry in enumerate(run_history):
        row_bg = C_LIGHT if idx % 2 == 0 else "white"
        _rect(cv, ML, y - 14, CW, 16, row_bg)

        ver = entry.get("version", "?")
        st = entry.get("status", "?")
        pm = entry.get("metrics", {})
        pm_val = f"{pm.get('value', 0):.4f}" if isinstance(pm, dict) and "value" in pm else str(pm)
        dp = entry.get("decision_path", "?") or "?"
        cl = entry.get("consensus_level", "?") or "?"
        gp = entry.get("gpt_suggested_path", "?") or "?"

        _text(cv, str(ver), col_x[0] + 3, y - 10, FN, 8, C_BLACK)
        st_color = C_SUCCESS if st == "success" else C_FAIL if st in ("failed", "smoke_failed") else C_WARN
        _text(cv, st, col_x[1] + 3, y - 10, FB, 8, st_color)
        _text(cv, pm_val, col_x[2] + 3, y - 10, FN, 8, C_BLACK)
        _text(cv, dp, col_x[3] + 3, y - 10, FB, 8, C_BLACK)
        _text(cv, cl, col_x[4] + 3, y - 10, FN, 8, C_BLACK)
        _text(cv, gp, col_x[5] + 3, y - 10, FN, 8, C_BLACK)

        y -= 16
        if y < 120:
            break

    y -= 10

    # ── 6. 분석 요약 ────────────────────────────────────
    if decision:
        _text(cv, "Final Decision", ML, y, FB, 11, C_HEADER)
        y -= 16
        _rect(cv, ML, y - 50, CW, 50, C_LIGHT, radius=4)

        reason = decision.get("decision_reason", decision.get("justification", ""))
        _text(cv, f"Path: {decision.get('path', '?')}", ML + 8, y - 14, FB, 10, C_BLACK)
        _text(cv, f"Evidence: {decision.get('evidence_strength', '?')} | Consensus: {decision.get('consensus_level', '?')}", ML + 8, y - 28, FN, 8, C_GRAY)
        _wrap_text(cv, reason[:200], ML + 8, y - 42, CW - 16, FN, 8, C_BLACK)
        y -= 60

    # ── 7. 훈련 안정성 ──────────────────────────────────
    stab = final_summary.get("training_stability", {})
    if stab:
        _text(cv, "Training Stability", ML, y, FB, 11, C_HEADER)
        y -= 16
        conv = stab.get("loss_converged", False)
        nan = stab.get("nan_detected", False)
        conv_icon = "Converged" if conv else "Not converged"
        nan_icon = "NaN detected" if nan else "Clean"
        _text(cv, f"Loss: {conv_icon}  |  NaN: {nan_icon}", ML + 8, y, FN, 9,
              C_SUCCESS if conv and not nan else C_WARN)
        y -= 20

    # ── 8. 푸터 ──────────────────────────────────────────
    _rect(cv, 0, 0, W, 25, C_BG)
    run_id = final_summary.get("run_id", "")
    _text(cv, f"Run: {run_id}", ML, 8, FN, 7, C_GRAY)
    _text(cv, f"Generated: {datetime.now().isoformat()}", W - ML - 130, 8, FN, 7, C_GRAY)

    cv.save()
    print(f"    [PDF] {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# research_loop에서 호출하는 편의 함수
# ──────────────────────────────────────────────────────────

def generate_result_report(
    pkg_dir: Path,
    final_summary: dict,
    run_history: list[dict],
    topic_file: str,
    hypothesis_file: str,
    decision: dict | None = None,
) -> Path | None:
    """research_loop 완료 후 결과 PDF를 생성한다.

    저장 위치: experiments/{slug}/results/result_report.pdf
    """
    try:
        topic = json.loads(Path(topic_file).read_text(encoding="utf-8"))
        hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    [PDF 경고] 입력 파일 읽기 실패: {e}")
        return None

    slug = slug_from_pkg(pkg_dir)
    res_dir = results_dir(slug)
    res_dir.mkdir(parents=True, exist_ok=True)
    output_path = res_dir / "result_report.pdf"

    try:
        return generate_result_pdf(
            final_summary=final_summary,
            run_history=run_history,
            topic=topic,
            hypothesis=hypothesis,
            decision=decision,
            output_path=output_path,
        )
    except Exception as e:
        print(f"    [PDF 경고] 결과 보고서 생성 실패: {e}")
        return None
