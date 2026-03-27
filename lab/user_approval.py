"""
Stage 5: 사용자 승인 인터페이스

전 단계 결과를 PDF 보고서로 생성하고 사용자의 승인/수정/거부를 받는다.
- PDF: experiments/{slug}/reports/report.pdf  (A4 최대 4페이지, decision memo)
- JSON: experiments/{slug}/reports/approval.json

PDF는 "10년차 박사 연구자 수준의 고밀도 연구 판단 문서"로 작성된다.
구조: 문제 정의 → evidence synthesis → 실험 변화·통찰 → 검증·결론 → references

사용법:
  python -m lab.user_approval \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \
    --papers-file     experiments/{slug}/reports/papers.json \
    --validation-file experiments/{slug}/reports/validation.json
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

# ── 상수 ──────────────────────────────────────────────────
from lab.config import SCORE_THRESHOLD as PDF_SCORE_THRESHOLD, topic_slug as _topic_slug, reports_dir as _reports_dir

MAX_PAGES = 4

C_BG      = "#1A1A2E"
C_HYPO    = "#0F3460"
C_ACCENT  = "#E94560"
C_GOLD    = "#F5A623"
C_APPROVE = "#27AE60"
C_WARN    = "#E67E22"
C_GRAY    = "#95A5A6"
C_CARD    = "#2C3E50"
C_LIGHT   = "#F0F4F8"


# ──────────────────────────────────────────────────────────
# 데이터 정규화 — flat/nested 구조 모두 처리
# ──────────────────────────────────────────────────────────

def _get(data: dict, key: str, fallback=None):
    """nested hypothesis dict → flat fallback 순서로 탐색."""
    nested = data.get("hypothesis", {})
    if isinstance(nested, dict):
        val = nested.get(key)
        if val:
            return val
    val = data.get(key)
    return val if val else (fallback if fallback is not None else "")


def _norm_hypothesis(h: dict) -> dict:
    """hypothesis.json을 PDF 렌더링용 통합 구조로 변환."""
    gap_raw = h.get("research_gap")
    if isinstance(gap_raw, dict) and gap_raw:
        gap = gap_raw
    else:
        gap = {
            "summary": h.get("research_gap_summary", ""),
            "limitations": h.get("limitations", []),
        }

    exp_raw = h.get("experiment_plan")
    if isinstance(exp_raw, dict) and exp_raw:
        exp = exp_raw
    else:
        exp = {
            "architecture": _get(h, "architecture"),
            "dataset": _get(h, "dataset"),
            "baseline_models": _get(h, "baseline_models", []),
            "evaluation_metrics": _get(h, "evaluation_metrics", []),
        }

    return {
        "statement_kr": _get(h, "statement_kr") or _get(h, "statement"),
        "statement": _get(h, "statement"),
        "key_innovation": _get(h, "key_innovation"),
        "expected_mechanism": _get(h, "expected_mechanism"),
        "rationale": _get(h, "rationale"),
        "research_gap": gap,
        "experiment_plan": exp,
        "risk_factors": _get(h, "risk_factors", []),
        "related_papers": _get(h, "related_papers", []),
        "falsification_criteria": _get(h, "falsification_criteria", []),
        "confidence": _get(h, "confidence", 0),
        "evidence_links": _get(h, "evidence_links", []),
    }


def _norm_validation(v: dict) -> dict:
    """validation.json을 PDF 렌더링용 통합 구조로 변환."""
    eval1 = v.get("gpt4o") or v.get("evaluator_1", {})
    eval2 = v.get("gemini") or v.get("evaluator_2", {})

    name1 = eval1.get("evaluator", "OpenAI")
    name2 = eval2.get("evaluator", "Gemini")
    if "claude" in name1.lower():
        name1 = "Claude-1"
    if "claude" in name2.lower():
        name2 = "Claude-2"

    return {
        "eval1": eval1,
        "eval2": eval2,
        "name1": name1,
        "name2": name2,
        "summary": v.get("summary", {}),
    }


# ──────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _setup_fonts():
    """reportlab 한글 폰트 등록. (regular, bold) 폰트명 반환."""
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
        bold    = next((m for m in matches if "Bold" in m or "bold" in m), regular)
        try:
            pdfmetrics.registerFont(TTFont("KR",      regular))
            pdfmetrics.registerFont(TTFont("KR-Bold", bold))
            return "KR", "KR-Bold"
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


# ──────────────────────────────────────────────────────────
# reportlab 드로잉 헬퍼
# ──────────────────────────────────────────────────────────

def _str_width(c, text: str, font: str, size: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    return stringWidth(text, font, size)


def _wrap_lines(c, text: str, max_w: float, font: str, size: float) -> list:
    """max_w 내에서 줄바꿈. 공백 기준 + 한국어 글자 단위 fallback."""
    words = str(text or "").split()
    lines, cur = [], ""
    for w in words:
        if _str_width(c, w, font, size) > max_w:
            if cur:
                lines.append(cur)
                cur = ""
            chunk = ""
            for ch in w:
                if _str_width(c, chunk + ch, font, size) > max_w:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
                else:
                    chunk += ch
            cur = chunk
            continue
        test = (cur + " " + w).strip()
        if _str_width(c, test, font, size) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _text_height(c, text: str, max_w: float, font: str, size: float,
                 line_h: float = None, max_lines: int = None) -> float:
    if line_h is None:
        line_h = size * 1.45
    lines = _wrap_lines(c, text, max_w, font, size)
    if max_lines:
        lines = lines[:max_lines]
    return len(lines) * line_h


def _color(color):
    from reportlab.lib.colors import HexColor, white, black
    if color == "white":
        return white
    if color == "black":
        return black
    return HexColor(color)


def _text(c, text: str, x: float, y: float, max_w: float,
          font: str, size: float, color: str,
          line_h: float = None, max_lines: int = None,
          align: str = "left") -> float:
    if line_h is None:
        line_h = size * 1.45
    lines = _wrap_lines(c, text, max_w, font, size)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    c.setFont(font, size)
    c.setFillColor(_color(color))
    curr_y = y
    for line in lines:
        if align == "center":
            lw = _str_width(c, line, font, size)
            c.drawString(x + (max_w - lw) / 2, curr_y, line)
        elif align == "right":
            lw = _str_width(c, line, font, size)
            c.drawString(x + max_w - lw, curr_y, line)
        else:
            c.drawString(x, curr_y, line)
        curr_y -= line_h
    return curr_y


def _rect(c, x: float, y: float, w: float, h: float,
          fill: str, radius: float = 0, stroke: str = None, stroke_w: float = 0.8):
    c.setFillColor(_color(fill))
    if stroke:
        c.setStrokeColor(_color(stroke))
        c.setLineWidth(stroke_w)
        do_stroke = 1
    else:
        c.setStrokeColor(_color(fill))
        do_stroke = 0
    if radius:
        c.roundRect(x, y, w, h, radius, fill=1, stroke=do_stroke)
    else:
        c.rect(x, y, w, h, fill=1, stroke=do_stroke)


def _truncate(c, text: str, font: str, size: float, max_w: float) -> str:
    """텍스트가 max_w를 초과하면 잘라서 ... 붙인다."""
    text = str(text or "")
    if _str_width(c, text, font, size) <= max_w:
        return text
    while text and _str_width(c, text + "...", font, size) > max_w:
        text = text[:-1]
    return text + "..."


# ── 일반화된 차트 시스템 ──────────────────────────────────
#
# 데이터 형태에 따라 차트 타입 자동 선택:
#   버전 ≤ 5, metric 1개  → bar chart
#   버전 ≤ 5, metric 2개+ → grouped bar chart
#   버전 > 5, metric 1개  → line chart
#   버전 > 5, metric 2개+ → multi-line chart
#   클래스별 accuracy      → horizontal bar (별도)

PALETTE = [C_HYPO, C_APPROVE, C_WARN, C_ACCENT, C_GOLD, C_GRAY,
           "#8E44AD", "#16A085", "#D35400", "#2980B9"]


def _detect_chart_metrics(run_history: list, primary_name: str) -> list[str]:
    """run_history에서 수치형 metric 키를 자동 감지한다."""
    skip = {"version", "epochs", "changes", "status"}
    if not run_history:
        return []
    keys = set()
    for e in run_history:
        for k, v in e.items():
            if k not in skip and isinstance(v, (int, float)):
                keys.add(k)
    result = []
    if primary_name in keys:
        result.append(primary_name)
        keys.discard(primary_name)
    result.extend(sorted(keys))
    return result


def _detect_class_metrics(all_metrics: dict) -> dict[str, float]:
    """all_metrics에서 클래스별 metric을 추출한다."""
    import re
    result = {}
    for k, v in all_metrics.items():
        m = re.match(r"class_(\w+)_(\w+)", k)
        if m and isinstance(v, (int, float)):
            label = f"Class {m.group(1)}"
            result[label] = v
    return result


def _auto_format(val: float, v_max: float) -> str:
    """값 범위에 따라 포맷을 자동 결정한다."""
    if v_max <= 2:
        return f"{val:.4f}"
    elif v_max <= 100:
        return f"{val:.2f}"
    else:
        return f"{val:.1f}"


def _chart_setup(c, x, y, w, h, all_vals, target, title, FN, FB):
    """차트 공통 셋업: 배경, 제목, y축, target 라인."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    _rect(c, x, y, w, h, "#F8F9FA", radius=4)

    PAD = 4 * mm
    Y_LABEL_W = 12 * mm
    chart_x = x + PAD + Y_LABEL_W
    chart_y = y + PAD + 6 * mm
    chart_w = w - 2 * PAD - Y_LABEL_W - 5 * mm
    chart_h = h - 2 * PAD - 14 * mm

    c.setFont(FB, 9)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(x + PAD, y + h - PAD - 3 * mm, title)

    if target and isinstance(target, (int, float)):
        all_vals = list(all_vals) + [target]
    v_min_raw = min(all_vals) if all_vals else 0
    v_max_raw = max(all_vals) if all_vals else 1
    margin = (v_max_raw - v_min_raw) * 0.1 or 0.005
    v_min = v_min_raw - margin
    v_max = v_max_raw + margin
    v_range = v_max - v_min if v_max > v_min else 0.01

    n_ticks = 4
    for i in range(n_ticks + 1):
        tick_val = v_min + v_range * i / n_ticks
        tick_y = chart_y + chart_h * i / n_ticks
        fmt = _auto_format(tick_val, v_max)
        c.setFont(FN, 5.5)
        c.setFillColor(HexColor(C_GRAY))
        c.drawRightString(chart_x - 1.5 * mm, tick_y - 1.5, fmt)
        c.setStrokeColor(HexColor("#E0E0E0"))
        c.setLineWidth(0.3)
        c.line(chart_x, tick_y, chart_x + chart_w, tick_y)

    if target and isinstance(target, (int, float)):
        target_y = chart_y + chart_h * (target - v_min) / v_range
        c.setStrokeColor(HexColor(C_ACCENT))
        c.setLineWidth(1)
        c.setDash(3, 2)
        c.line(chart_x, target_y, chart_x + chart_w, target_y)
        c.setDash()
        c.setFont(FN, 5.5)
        c.setFillColor(HexColor(C_ACCENT))
        c.drawString(chart_x + chart_w + 1 * mm, target_y - 1.5, "target")

    return chart_x, chart_y, chart_w, chart_h, v_min, v_max, v_range


def _chart_legend(c, metric_keys, chart_x, legend_y, FN):
    """범례를 그린다."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    lx = chart_x
    for mi, mk in enumerate(metric_keys):
        color = PALETTE[mi % len(PALETTE)]
        _rect(c, lx, legend_y, 3 * mm, 3 * mm, color, radius=1)
        lbl = mk.replace("_", " ")
        c.setFont(FN, 6)
        c.setFillColor(HexColor("#444444"))
        c.drawString(lx + 4 * mm, legend_y + 0.5 * mm, lbl)
        lx += _str_width(c, lbl, FN, 6) + 8 * mm


def _chart_bar(c, cx, cy, cw, ch, v_min, v_max, v_range,
               run_history, metric_keys, target, FN, FB):
    """Bar chart / Grouped bar chart."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    n = len(run_history)
    nm = len(metric_keys)
    group_w = cw / n
    bar_w = min(group_w * 0.7 / nm, 18 * mm)
    group_pad = (group_w - bar_w * nm) / 2

    for vi, entry in enumerate(run_history):
        gx = cx + vi * group_w + group_pad
        for mi, mk in enumerate(metric_keys):
            val = entry.get(mk, 0)
            if not isinstance(val, (int, float)):
                continue
            bx = gx + mi * bar_w
            bh = ch * (val - v_min) / v_range
            color = PALETTE[mi % len(PALETTE)]
            if nm == 1:
                color = C_APPROVE if target and val >= target else C_WARN
            _rect(c, bx, cy, bar_w, max(bh, 1), color, radius=2)

            if nm <= 2:
                fmt = _auto_format(val, v_max)
                c.setFont(FB, 5.5)
                c.setFillColor(HexColor("#333333"))
                lw = _str_width(c, fmt, FB, 5.5)
                c.drawString(bx + (bar_w - lw) / 2, cy + bh + 1 * mm, fmt)

        lbl = f"v{entry.get('version', vi+1)}"
        c.setFont(FN, 7)
        c.setFillColor(HexColor("#555555"))
        lw = _str_width(c, lbl, FN, 7)
        c.drawString(gx + (bar_w * nm - lw) / 2, cy - 4 * mm, lbl)


def _chart_line(c, cx, cy, cw, ch, v_min, v_range,
                run_history, metric_keys, FN, FB):
    """Line chart / Multi-line chart."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    n = len(run_history)
    if n < 2:
        return

    for mi, mk in enumerate(metric_keys):
        color = PALETTE[mi % len(PALETTE)]
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.8)

        points = []
        for vi, entry in enumerate(run_history):
            val = entry.get(mk, 0)
            if not isinstance(val, (int, float)):
                continue
            px = cx + (vi / (n - 1)) * cw
            py = cy + ch * (val - v_min) / v_range
            points.append((px, py, val))

        if len(points) >= 2:
            p = c.beginPath()
            p.moveTo(points[0][0], points[0][1])
            for px, py, _ in points[1:]:
                p.lineTo(px, py)
            c.drawPath(p, fill=0, stroke=1)

        for pi, (px, py, val) in enumerate(points):
            c.setFillColor(HexColor(color))
            c.circle(px, py, 2.5, fill=1, stroke=0)

            if len(metric_keys) <= 2:
                fmt = _auto_format(val, max(v[2] for v in points))
                c.setFont(FB, 5.5)
                c.setFillColor(HexColor("#333333"))
                c.drawString(px + 2 * mm, py + 1.5 * mm, fmt)

    for vi, entry in enumerate(run_history):
        px = cx + (vi / max(n - 1, 1)) * cw
        lbl = f"v{entry.get('version', vi+1)}"
        c.setFont(FN, 7)
        c.setFillColor(HexColor("#555555"))
        lw = _str_width(c, lbl, FN, 7)
        c.drawString(px - lw / 2, cy - 4 * mm, lbl)


def _chart_hbar(c, x, y, w, h, categories: dict[str, float],
                target, title, FN, FB):
    """Horizontal bar chart — 카테고리별 비교."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    if not categories:
        return

    _rect(c, x, y, w, h, "#F8F9FA", radius=4)

    PAD = 4 * mm
    LABEL_W = 20 * mm

    c.setFont(FB, 9)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(x + PAD, y + h - PAD - 3 * mm, title)

    items = list(categories.items())
    n = len(items)
    chart_x = x + PAD + LABEL_W
    chart_w = w - 2 * PAD - LABEL_W - 10 * mm
    chart_top = y + h - PAD - 10 * mm
    avail_h = chart_top - y - PAD
    bar_h = min(avail_h / n * 0.7, 5 * mm)
    bar_gap = (avail_h - bar_h * n) / max(n, 1)

    vals = [v for _, v in items]
    v_max = max(max(vals), target or 0) if vals else 1
    v_min = min(vals) if vals else 0
    bar_scale = v_max * 1.05 if v_max > 0 else 1

    if target and isinstance(target, (int, float)):
        tx = chart_x + chart_w * (target / bar_scale)
        c.setStrokeColor(HexColor(C_ACCENT))
        c.setLineWidth(1)
        c.setDash(3, 2)
        c.line(tx, y + PAD, tx, chart_top)
        c.setDash()
        c.setFont(FN, 5)
        c.setFillColor(HexColor(C_ACCENT))
        c.drawString(tx + 1 * mm, chart_top + 1 * mm, f"target {target}")

    for i, (label, val) in enumerate(items):
        by = chart_top - (i + 1) * (bar_h + bar_gap) + bar_gap / 2
        bw = chart_w * (val / bar_scale)
        color = C_APPROVE if target and val >= target else C_WARN if val >= v_min + (v_max - v_min) * 0.5 else C_ACCENT

        c.setFont(FN, 6)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(chart_x - 2 * mm, by + bar_h / 2 - 2, label)

        _rect(c, chart_x, by, max(bw, 1), bar_h, color, radius=2)

        fmt = _auto_format(val, v_max)
        c.setFont(FB, 5.5)
        c.setFillColor(HexColor("#333333"))
        c.drawString(chart_x + bw + 1.5 * mm, by + bar_h / 2 - 2, fmt)


def _draw_chart(c, x, y, w, h, run_history, metric_keys, target,
                primary_name, FN, FB):
    """데이터 형태에 따라 차트 타입을 자동 선택하여 그린다."""
    from reportlab.lib.units import mm

    if not run_history or not metric_keys:
        return

    all_vals = []
    for e in run_history:
        for k in metric_keys:
            v = e.get(k, 0)
            if isinstance(v, (int, float)):
                all_vals.append(v)

    n_versions = len(run_history)
    n_metrics = len(metric_keys)
    use_line = n_versions > 5

    if n_metrics == 1:
        title = f"{primary_name.replace('_', ' ').title()} Progression"
    else:
        title = "Metrics Comparison"

    legend_h = 6 * mm if n_metrics > 1 else 0
    chart_h_actual = h - legend_h

    cx, cy, cw, ch, v_min, v_max, v_range = _chart_setup(
        c, x, y + legend_h, w, chart_h_actual, all_vals, target, title, FN, FB)

    if use_line:
        _chart_line(c, cx, cy, cw, ch, v_min, v_range,
                    run_history, metric_keys, FN, FB)
    else:
        _chart_bar(c, cx, cy, cw, ch, v_min, v_max, v_range,
                   run_history, metric_keys, target, FN, FB)

    if n_metrics > 1:
        _chart_legend(c, metric_keys, cx, y + 1 * mm, FN)


# ── Reference formatting ──────────────────────────────────

def _format_reference(paper: dict, idx: int) -> str:
    """논문을 학술 참고문헌 형식으로 포맷한다."""
    authors = paper.get("authors", [])
    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + ", et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Unknown"

    title = paper.get("title", "Untitled")
    venue = paper.get("venue", paper.get("source", ""))
    year  = paper.get("year", "")

    ref = f"[{idx}] {author_str}. \"{title}.\""
    if venue:
        ref += f" {venue},"
    if year:
        ref += f" {year}."
    elif ref.endswith(","):
        ref = ref[:-1] + "."
    return ref


# ══════════════════════════════════════════════════════════
# Decision Memo — 고밀도 연구 판단 문서 (max 4 pages)
# ══════════════════════════════════════════════════════════

# 렌더링 상수
_TEXT_SIZE   = 7.5
_TEXT_LH     = 10.5
_BULLET_SIZE = 7.0
_BULLET_LH   = 9.5
_REF_SIZE    = 6.5
_REF_LH      = 9.0
_SUB_SIZE    = 8.0
_SEC_SIZE    = 9.0


def _draw_memo_header(c, W, H, FN, FB, title, page_num):
    """Minimal memo header: title + date + page number."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    HDR_Y = H - 10 * mm
    c.setStrokeColor(HexColor(C_HYPO))
    c.setLineWidth(1.5)
    c.line(10 * mm, HDR_Y, W - 10 * mm, HDR_Y)

    c.setFont(FB, 12)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(10 * mm, HDR_Y + 3 * mm, title)

    c.setFont(FN, 7)
    c.setFillColor(HexColor(C_GRAY))
    date_str = datetime.now().strftime("%Y-%m-%d")
    c.drawRightString(W - 10 * mm, HDR_Y + 3 * mm, f"{date_str}  |  p.{page_num}")

    return HDR_Y - 5 * mm


def _draw_section_heading(c, text, x, y, w, FB, size=None):
    """Draw section heading with underline."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    if size is None:
        size = _SEC_SIZE
    c.setFont(FB, size)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(x, y, text)
    c.setStrokeColor(HexColor("#D0D0D0"))
    c.setLineWidth(0.5)
    c.line(x, y - 2, x + w, y - 2)
    return y - size - 3 * mm


# ──────────────────────────────────────────────────────────
# Section content builders
#
# 각 builder는 렌더링 항목 리스트를 반환한다.
# 항목 형식: (type, content, ...)
#   ("text", str)              — 본문 문단
#   ("bullet", str)            — 불릿 항목
#   ("kv", label, value)       — key: value (label 볼드)
#   ("subheading", str)        — 소제목 (볼드)
#   ("gap", float)             — 수직 간격 (mm)
#   ("ref", str)               — 참고문헌 항목 (작은 폰트)
#   ("chart_placeholder", _)   — 차트 위치 표시
# ──────────────────────────────────────────────────────────

def _strength_score(paper: dict) -> int:
    """support_strength를 수치로 변환 (정렬용)."""
    s = paper.get("support_strength", "").lower()
    return {"strong": 3, "moderate": 2, "medium": 2, "weak": 1}.get(s, 0)


def _build_section1(inp: dict, hyp: dict) -> list:
    """Section 1: Problem Definition & Hypothesis Summary."""
    items = []

    problem = inp.get("problem_definition", "")
    if problem:
        items.append(("kv", "Problem", problem))

    constraints = inp.get("constraints", "")
    if constraints:
        items.append(("kv", "Constraints", constraints))

    target = inp.get("target_metric", "")
    outcome = inp.get("desired_outcome", "")
    target_text = target
    if outcome:
        target_text += f" — {outcome}"
    if target_text:
        items.append(("kv", "Target metric", target_text))

    items.append(("gap", 3))

    stmt = hyp.get("statement_kr") or hyp.get("statement", "")
    if stmt:
        items.append(("kv", "Hypothesis", stmt))

    innov = hyp.get("key_innovation", "")
    if innov:
        items.append(("kv", "Novelty", innov))

    mech = hyp.get("expected_mechanism", "")
    if mech:
        items.append(("kv", "Mechanism", mech))

    return items


def _build_section2(hyp: dict, papers_list: list, val_norm: dict,
                    decisive_evidence: dict | None = None) -> list:
    """Section 2: Related Work & Evidence Synthesis."""
    items = []

    # Top papers by rank / support_strength
    ranked = sorted(papers_list,
                    key=lambda p: (p.get("rank", 99), -_strength_score(p)))
    top_papers = ranked[:5]

    for i, p in enumerate(top_papers):
        role = p.get("evidence_role", "")
        slots = p.get("claim_slots_supported", [])
        why = p.get("why_selected", "")
        strength = p.get("support_strength", "")

        authors = p.get("authors", [])
        author_short = authors[0].split(",")[0].split()[-1] if authors else "?"
        year = p.get("year", "")

        line = f"[{i+1}] {author_short} et al."
        if year:
            line += f" ({year})"
        line += f": {role}" if role else ""
        if slots:
            line += f". Supports {', '.join(str(s) for s in slots[:3])}"
        if strength:
            line += f" (strength: {strength})"
        line += "."
        if why:
            line += f" {why}"
        items.append(("bullet", line))

    items.append(("gap", 2))

    # Research gap / closest prior art
    gap = hyp.get("research_gap", {})
    gap_summary = gap.get("summary", "") if isinstance(gap, dict) else str(gap)
    if gap_summary:
        items.append(("kv", "Research gap", gap_summary))

    lims = gap.get("limitations", []) if isinstance(gap, dict) else []
    for lim in lims[:3]:
        items.append(("bullet", str(lim)))

    items.append(("gap", 2))

    # Weaknesses / competing explanations from validation
    summary = val_norm.get("summary", {})
    weaknesses = summary.get("all_weaknesses", [])
    if weaknesses:
        items.append(("subheading", "Key Weaknesses / Competing Explanations"))
        for w in weaknesses[:3]:
            items.append(("bullet", str(w)))

    # Falsification criteria
    falsif = hyp.get("falsification_criteria", [])
    if falsif:
        items.append(("gap", 2))
        items.append(("subheading", "Falsification Criteria"))
        for fc in falsif[:4]:
            items.append(("bullet", str(fc) if isinstance(fc, str) else str(fc)))

    # Decisive Evidence (판세를 바꾸는 핵심 근거)
    if decisive_evidence:
        de_support = decisive_evidence.get("support", [])
        de_contra = decisive_evidence.get("contra", [])
        de_swing = decisive_evidence.get("swing", [])
        if de_support or de_contra or de_swing:
            items.append(("gap", 2))
            items.append(("subheading", "Decisive Evidence"))
            for s in de_support[:2]:
                if s.get("paper_id"):
                    items.append(("bullet", f"Support ({s.get('supports_dimension','')}): {s.get('title','')} — {s.get('reason','')[:80]}"))
            for c in de_contra[:2]:
                if c.get("paper_id"):
                    items.append(("bullet", f"Contra ({c.get('threatens_dimension','')}): {c.get('title','')} — {c.get('reason','')[:80]}"))
            for w in de_swing[:1]:
                if w.get("paper_id"):
                    items.append(("bullet", f"Swing: {w.get('title','')} — if confirmed: {w.get('if_confirmed_changes','')}"))

    return items


def _build_section3(run_history: list, final_summary: dict,
                    primary_name: str, hyp: dict) -> list:
    """Section 3: Experiment Changes & Insights."""
    items = []
    if not run_history:
        return items

    primary = final_summary.get("primary_metric", {})
    target = primary.get("target", 0)
    met = primary.get("met", False)
    n_runs = len(run_history)
    first_val = run_history[0].get(primary_name, 0)
    last_val = run_history[-1].get(primary_name, 0)
    total_delta = last_val - first_val

    # Overview line
    overview = f"{primary_name}: {first_val:.4f} → {last_val:.4f}"
    if total_delta != 0:
        sign = "+" if total_delta > 0 else ""
        overview += f" ({sign}{total_delta:.4f})"
    overview += f" over {n_runs} run(s). Target: {target}. {'Met.' if met else 'Not met.'}"
    items.append(("text", overview))

    items.append(("gap", 1))
    items.append(("chart_placeholder", None))
    items.append(("gap", 2))

    # Experiment Changes
    items.append(("subheading", "Experiment Changes"))
    for i, entry in enumerate(run_history):
        val = entry.get(primary_name, 0)
        changes = entry.get("changes", "")
        params = entry.get("params_M", "?")
        ver = entry.get("version", i + 1)

        if i == 0:
            line = f"v{ver} (baseline): {primary_name}={val:.4f}, {params}M params. {changes}"
        else:
            prev_val = run_history[i - 1].get(primary_name, 0)
            delta = val - prev_val
            sign = "+" if delta >= 0 else ""
            pct = (delta / prev_val * 100) if prev_val else 0
            effect = f"{sign}{delta:.4f} ({sign}{pct:.1f}%)" if delta != 0 else "no change"
            line = f"v{ver}: {changes} → {effect}. {primary_name}={val:.4f}, {params}M params."
        items.append(("bullet", line))

    items.append(("gap", 2))

    # Observed Effects
    items.append(("subheading", "Observed Effects"))

    stab = final_summary.get("training_stability", {})
    if stab:
        conv = stab.get("loss_converged", False)
        nan_det = stab.get("nan_detected", False)
        notes = stab.get("notes", [])
        stab_line = f"Stability: {'converged' if conv else 'not converged'}, NaN: {'detected' if nan_det else 'clean'}."
        if notes:
            typed_notes = [str(n) for n in notes[:3]]
            stab_line += f" ({'; '.join(typed_notes)})"
        items.append(("bullet", stab_line))

    # Constraint satisfaction
    hyp_impl = final_summary.get("hypothesis_implementation", {})
    cons_audit = hyp_impl.get("constraints_audit", {}) if hyp_impl else {}
    if cons_audit:
        violations = cons_audit.get("violations", [])
        n_recognized = len(cons_audit.get("recognized_constraints", []))
        if cons_audit.get("implemented"):
            items.append(("bullet", f"Constraints: {n_recognized} recognized, all satisfied."))
        else:
            items.append(("bullet", f"Constraints: {n_recognized} recognized, {len(violations)} violation(s): {'; '.join(str(v) for v in violations[:2])}."))

    # Effect size (vs previous version)
    effect_size = final_summary.get("effect_size", {})
    if effect_size and effect_size.get("magnitude"):
        items.append(("bullet", f"Effect size: {effect_size['interpretation']}"))

    # Metric validation warnings
    metric_warnings = final_summary.get("metric_warnings", [])
    if metric_warnings:
        items.append(("bullet", f"Metric warnings: {'; '.join(metric_warnings[:3])}"))

    # Secondary metrics
    sec = final_summary.get("secondary_metrics", [])
    if sec:
        sec_str = ", ".join(f"{m.get('name','?')}={m.get('value',0):.4f}" for m in sec[:4])
        items.append(("bullet", f"Secondary metrics: {sec_str}."))

    items.append(("gap", 2))

    # Insights
    items.append(("subheading", "Insights"))

    bottlenecks = final_summary.get("bottlenecks", [])
    if bottlenecks:
        for bn in bottlenecks[:3]:
            cat = bn.get("category", "")
            detail = bn.get("detail", "")
            conf = bn.get("confidence", "")
            line = f"{cat}: {detail}"
            if conf:
                line += f" (confidence: {conf})"
            items.append(("bullet", line))

    # Mechanism audit insight
    mech_audit = hyp_impl.get("mechanism_audit", {}) if hyp_impl else {}
    if mech_audit:
        impl = mech_audit.get("implemented", False)
        risk = mech_audit.get("risk_level", "")
        if impl:
            evidence = mech_audit.get("evidence", [])
            ev_str = f" Evidence: {evidence[0]}" if evidence else ""
            items.append(("bullet", f"Core mechanism: implemented (risk: {risk}).{ev_str}"))
        else:
            missing = mech_audit.get("missing_links", [])
            items.append(("bullet", f"Core mechanism: NOT implemented (risk: {risk}). Missing: {'; '.join(str(m) for m in missing[:2])}."))

    # Ablation / revision insights
    ablation = final_summary.get("ablation_findings", [])
    if ablation:
        for ab in ablation[:2]:
            desc = ab.get("description", "") if isinstance(ab, dict) else str(ab)
            items.append(("bullet", desc))

    return items


def _build_section4(run_history: list, final_summary: dict,
                    primary_name: str, hyp: dict, val_norm: dict,
                    has_results: bool) -> list:
    """Section 4: Hypothesis Verification & Conclusion."""
    items = []

    summary = val_norm.get("summary", {})
    eval1 = val_norm.get("eval1", {})
    eval2 = val_norm.get("eval2", {})
    name1 = val_norm.get("name1", "Evaluator 1")
    name2 = val_norm.get("name2", "Evaluator 2")

    score1 = eval1.get("score", 0)
    score2 = eval2.get("score", 0)
    verdict1 = eval1.get("verdict", "")
    verdict2 = eval2.get("verdict", "")
    avg = summary.get("average_score", 0)
    overall = summary.get("overall_verdict", "")

    items.append(("text",
        f"Validation: {name1} {score1}/10 [{verdict1}], "
        f"{name2} {score2}/10 [{verdict2}]. "
        f"Average: {avg}/10 [{overall}]."))

    items.append(("gap", 2))

    # Novelty
    innov = hyp.get("key_innovation", "")
    strengths = summary.get("all_strengths", [])
    novelty_text = innov
    if strengths:
        novelty_text += f" Strengths: {'; '.join(str(s) for s in strengths[:2])}."
    items.append(("kv", "Novelty", novelty_text))

    # Feasibility
    risks = hyp.get("risk_factors", [])
    feasibility_text = ""
    if risks:
        feasibility_text = f"Risk factors: {'; '.join(str(r) for r in risks[:3])}."

    exp_plan = hyp.get("experiment_plan", {})
    if isinstance(exp_plan, dict):
        arch = exp_plan.get("architecture", "")
        if arch:
            feasibility_text = f"Architecture: {arch}. " + feasibility_text
    items.append(("kv", "Feasibility", feasibility_text or "No specific risks identified."))

    # Falsification
    falsif = hyp.get("falsification_criteria", [])
    if falsif:
        if has_results and run_history:
            primary = final_summary.get("primary_metric", {})
            met = primary.get("met", False)
            last_val = run_history[-1].get(primary_name, 0)
            target_val = primary.get("target", 0)
            items.append(("kv", "Falsification",
                f"Primary: {'PASS' if met else 'FAIL'} "
                f"({primary_name}={last_val:.4f} vs target {target_val})."))
        else:
            items.append(("kv", "Falsification",
                f"{len(falsif)} criteria defined. Awaiting experiment."))

    # Implementation audit
    if has_results and final_summary:
        hyp_impl = final_summary.get("hypothesis_implementation", {})
        if hyp_impl:
            mech_ok = hyp_impl.get("mechanism_audit", {}).get("implemented", "?")
            metr_ok = hyp_impl.get("metric_audit", {}).get("implemented", "?")
            cons_ok = hyp_impl.get("constraints_audit", {}).get("implemented", "?")

            def _pass_fail(v):
                if v is True:
                    return "PASS"
                if v is False:
                    return "FAIL"
                return "?"

            items.append(("text",
                f"Implementation audit — mechanism: {_pass_fail(mech_ok)}, "
                f"metric: {_pass_fail(metr_ok)}, "
                f"constraints: {_pass_fail(cons_ok)}."))

    # Scientific Bet + Problem Reframe (A-7)
    if has_results and final_summary:
        bet = final_summary.get("scientific_bet", {})
        if bet and bet.get("bet_grade"):
            items.append(("gap", 2))
            items.append(("kv", "Scientific bet",
                f"Grade {bet['bet_grade']}. {bet.get('claim', '')} "
                f"{bet.get('hedge', '')}"))

        reframe = final_summary.get("problem_reframe", {})
        if reframe and reframe.get("reframe_detected"):
            items.append(("gap", 1))
            items.append(("text",
                f"Problem reframe signal detected ({len(reframe.get('signals',[]))} signals): "
                f"{reframe.get('suggested_reframe', '')}"))

        pivots = final_summary.get("pivotal_evidence", [])
        if pivots:
            items.append(("gap", 1))
            items.append(("subheading", "Pivotal Evidence"))
            for p in pivots[:3]:
                items.append(("bullet", f"{p.get('finding','')} — {p.get('why_pivotal','')} → {p.get('action','')}"))

    items.append(("gap", 3))

    # A. Conclusion
    items.append(("subheading", "Conclusion"))
    if has_results and run_history:
        primary = final_summary.get("primary_metric", {})
        met = primary.get("met", False)
        n_runs = len(run_history)
        first_val = run_history[0].get(primary_name, 0)
        last_val = run_history[-1].get(primary_name, 0)
        target_val = primary.get("target", 0)

        if met:
            items.append(("text",
                f"Over {n_runs} iteration(s), {primary_name} improved from "
                f"{first_val:.4f} to {last_val:.4f}, exceeding the target of "
                f"{target_val}. The proposed mechanism is validated under current "
                f"constraints. Strongest competing explanation should be re-evaluated "
                f"before claiming generality beyond the tested regime."))
        else:
            gap_val = target_val - last_val
            items.append(("text",
                f"After {n_runs} iteration(s), {primary_name} reached {last_val:.4f} "
                f"(target: {target_val}, gap: {gap_val:.4f}). The core direction "
                f"shows marginal progress but has not met the success criterion. "
                f"The bottleneck should be identified before further iterations."))

        confidence = final_summary.get("confidence", {})
        if isinstance(confidence, dict) and confidence.get("overall"):
            items.append(("text",
                f"Confidence: {confidence['overall']:.2f}. "
                f"{confidence.get('explanation', '')}"))
    else:
        items.append(("text",
            f"Pre-experiment validation: {overall} (avg {avg}/10). "))
        weaknesses = summary.get("all_weaknesses", [])
        if weaknesses:
            items.append(("text", f"Primary concern: {weaknesses[0]}"))

    items.append(("gap", 2))

    # B. Final Recommendation
    items.append(("subheading", "Final Recommendation"))
    if has_results and final_summary:
        rec_actions = final_summary.get("recommended_next_actions", [])
        if rec_actions:
            top = rec_actions[0] if isinstance(rec_actions[0], dict) else {
                "path": "?", "rationale": str(rec_actions[0])}
            path = top.get("path", "?")
            rationale = top.get("rationale", "")
            items.append(("text", f"Path {path}: {rationale}"))
        else:
            primary = final_summary.get("primary_metric", {})
            met = primary.get("met", False)
            if met:
                items.append(("text", "Target met. Proceed to finalize or extend to validation set."))
            else:
                items.append(("text", "Target not met. Path A recommended: iterate on implementation."))
    else:
        if overall == "approve":
            items.append(("text", "Approve: proceed to experiment to validate hypothesis."))
        elif overall == "reject":
            items.append(("text", "Reject: hypothesis does not pass validation threshold. Revise or abandon."))
        else:
            items.append(("text", f"{overall.title()}: address identified weaknesses before proceeding."))

    items.append(("gap", 2))

    # C. Next Actions
    items.append(("subheading", "Next Actions"))
    if has_results and final_summary:
        rec_actions = final_summary.get("recommended_next_actions", [])
        for ra in rec_actions[:3]:
            if isinstance(ra, dict):
                text = ra.get("rationale", "")
                priority = ra.get("priority", "")
                if priority:
                    text += f" (priority: {priority})"
            else:
                text = str(ra)
            items.append(("bullet", text))
        if not rec_actions:
            items.append(("bullet", "Review diagnostic output and determine next experiment direction."))
    else:
        items.append(("bullet", "Run initial experiment to validate hypothesis mechanism."))
        weaknesses = summary.get("all_weaknesses", [])
        if weaknesses:
            items.append(("bullet", f"Address before experiment: {weaknesses[0]}"))
        items.append(("bullet", "Confirm data availability and constraint feasibility."))

    return items


def _build_section5(papers_list: list) -> list:
    """Section 5: References — key 3-8 papers with relevance note."""
    items = []

    ranked = sorted(papers_list,
                    key=lambda p: (p.get("rank", 99), -_strength_score(p)))
    key_papers = ranked[:8]

    for i, p in enumerate(key_papers):
        ref = _format_reference(p, i + 1)
        role = p.get("evidence_role", "")
        slots = p.get("claim_slots_supported", [])
        note = ""
        if role:
            note = f" — {role}"
        elif slots:
            note = f" — supports {', '.join(str(s) for s in slots[:2])}"
        items.append(("ref", ref + note))

    return items


# ──────────────────────────────────────────────────────────
# Unified memo renderer
# ──────────────────────────────────────────────────────────

def _render_decision_memo(c, W, H, FN, FB, memo_title, sections,
                          chart_data=None):
    """Decision memo를 flowing text로 렌더링한다 (최대 MAX_PAGES 페이지).

    Args:
        sections: list of (section_title, items) — 각 섹션과 렌더링 항목
        chart_data: optional dict — 인라인 차트 데이터
    """
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    ML = 10 * mm
    CW = W - 2 * ML
    BOTTOM = 12 * mm

    page_num = 1
    cur_y = _draw_memo_header(c, W, H, FN, FB, memo_title, page_num)

    def _break(needed_h):
        """공간 부족 시 페이지 넘김. 성공 여부 반환."""
        nonlocal cur_y, page_num
        if cur_y - needed_h >= BOTTOM:
            return True
        if page_num >= MAX_PAGES:
            return False
        c.showPage()
        page_num += 1
        cur_y = _draw_memo_header(c, W, H, FN, FB, memo_title, page_num)
        return True

    for sec_title, items in sections:
        # Section heading
        if not _break(14 * mm):
            break
        cur_y = _draw_section_heading(c, sec_title, ML, cur_y, CW, FB)
        cur_y -= 1 * mm

        for item in items:
            item_type = item[0]

            if item_type == "gap":
                gap_val = item[1] if len(item) > 1 and isinstance(item[1], (int, float)) else 2
                cur_y -= gap_val * mm

            elif item_type == "text":
                content = item[1] or ""
                h = _text_height(c, content, CW, FN, _TEXT_SIZE, line_h=_TEXT_LH)
                if not _break(h + 2 * mm):
                    break
                cur_y = _text(c, content, ML, cur_y, CW,
                              FN, _TEXT_SIZE, "#333333", line_h=_TEXT_LH)
                cur_y -= 2 * mm

            elif item_type == "bullet":
                content = item[1] or ""
                btext = f"\u2022 {content}"
                h = _text_height(c, btext, CW - 4 * mm, FN, _BULLET_SIZE, line_h=_BULLET_LH)
                if not _break(h + 1.5 * mm):
                    break
                cur_y = _text(c, btext, ML + 3 * mm, cur_y, CW - 4 * mm,
                              FN, _BULLET_SIZE, "#333333", line_h=_BULLET_LH)
                cur_y -= 1.5 * mm

            elif item_type == "kv":
                label = item[1] if len(item) > 1 else ""
                value = item[2] if len(item) > 2 else ""
                label_str = f"{label}: "
                full_text = label_str + value
                h = _text_height(c, full_text, CW, FN, _TEXT_SIZE, line_h=_TEXT_LH)
                if not _break(h + 2 * mm):
                    break

                # label bold, value regular
                label_w = _str_width(c, label_str, FB, _TEXT_SIZE)
                # label_w를 전체 폭의 25% 이내로 제한
                max_label_w = CW * 0.25
                if label_w > max_label_w:
                    # label이 너무 길면 한 줄 전체를 text로 렌더링
                    cur_y = _text(c, full_text, ML, cur_y, CW,
                                  FN, _TEXT_SIZE, "#333333", line_h=_TEXT_LH)
                else:
                    c.setFont(FB, _TEXT_SIZE)
                    c.setFillColor(HexColor(C_HYPO))
                    c.drawString(ML, cur_y, label_str)
                    if value:
                        cur_y = _text(c, value, ML + label_w, cur_y,
                                      CW - label_w, FN, _TEXT_SIZE, "#333333",
                                      line_h=_TEXT_LH)
                    else:
                        cur_y -= _TEXT_LH
                cur_y -= 2 * mm

            elif item_type == "subheading":
                content = item[1] or ""
                if not _break(6 * mm):
                    break
                c.setFont(FB, _SUB_SIZE)
                c.setFillColor(HexColor(C_CARD))
                c.drawString(ML, cur_y, content)
                cur_y -= _SUB_SIZE + 2 * mm

            elif item_type == "ref":
                content = item[1] or ""
                h = _text_height(c, content, CW, FN, _REF_SIZE, line_h=_REF_LH)
                if not _break(h + 1.5 * mm):
                    break
                cur_y = _text(c, content, ML, cur_y, CW,
                              FN, _REF_SIZE, "#444444", line_h=_REF_LH)
                cur_y -= 1.5 * mm

            elif item_type == "chart_placeholder" and chart_data:
                CHART_H = 38 * mm
                if not _break(CHART_H + 4 * mm):
                    break
                rh = chart_data.get("run_history", [])
                mk = chart_data.get("metric_keys", [])
                tgt = chart_data.get("target", 0)
                pn = chart_data.get("primary_name", "")
                cc = chart_data.get("class_cats", {})

                if cc:
                    LW = CW * 0.55
                    RW = CW - LW - 3 * mm
                    _draw_chart(c, ML, cur_y - CHART_H, LW, CHART_H,
                                rh, mk, tgt, pn, FN, FB)
                    _chart_hbar(c, ML + LW + 3 * mm, cur_y - CHART_H,
                                RW, CHART_H, cc, tgt, "Per-Class", FN, FB)
                elif rh and mk:
                    _draw_chart(c, ML, cur_y - CHART_H, CW, CHART_H,
                                rh, mk, tgt, pn, FN, FB)
                cur_y -= CHART_H + 4 * mm
        else:
            # items 루프가 정상 종료 → 다음 섹션으로
            continue
        # items 루프가 break로 종료 → 페이지 한도 도달
        break

    # 푸터
    c.setStrokeColor(HexColor(C_GRAY))
    c.setLineWidth(0.3)
    c.line(ML, 8 * mm, W - ML, 8 * mm)
    c.setFont(FN, 5.5)
    c.setFillColor(HexColor(C_GRAY))
    c.drawString(ML, 4 * mm, "AI-Driven Research Automation  |  Generated by Claude")
    c.drawRightString(W - ML, 4 * mm, datetime.now().strftime("%Y-%m-%d %H:%M"))

    return page_num


# ──────────────────────────────────────────────────────────
# PDF 생성 (통합)
# ──────────────────────────────────────────────────────────

def generate_pdf(
    topic: dict,
    hypothesis: dict,
    papers: dict,
    validation: dict,
    output_path: Path,
    results: list[dict] | None = None,
    final_summary: dict | None = None,
) -> None:
    """A4 최대 4페이지 Decision Memo PDF 생성.

    구조:
      1. Problem Definition & Hypothesis
      2. Related Work & Evidence Synthesis
      3. Experiment Changes & Insights (결과 있을 때만)
      4. Hypothesis Verification & Conclusion
      5. References
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4

    FN, FB = _setup_fonts()
    W, H   = A4
    output_path.parent.mkdir(exist_ok=True)
    cv = rl_canvas.Canvas(str(output_path), pagesize=A4)

    # 데이터 정규화
    inp         = topic.get("input", {})
    hyp         = _norm_hypothesis(hypothesis)
    val_n       = _norm_validation(validation)
    papers_list = papers.get("papers", [])
    has_results = results is not None and final_summary is not None

    # 섹션 빌드
    sec_num = 1
    sections = []

    sections.append((f"{sec_num}. Problem Definition & Hypothesis",
                     _build_section1(inp, hyp)))
    sec_num += 1

    decisive_ev = papers.get("decisive_evidence")
    sections.append((f"{sec_num}. Related Work & Evidence Synthesis",
                     _build_section2(hyp, papers_list, val_n, decisive_ev)))
    sec_num += 1

    chart_data = None
    if has_results:
        primary_name = final_summary.get("primary_metric", {}).get("name", "accuracy")
        s3_items = _build_section3(results, final_summary, primary_name, hyp)
        sections.append((f"{sec_num}. Experiment Changes & Insights", s3_items))
        sec_num += 1

        # 차트 데이터 준비
        chart_metrics = _detect_chart_metrics(results, primary_name)[:4]
        all_m = final_summary.get("all_metrics", {})
        class_cats = _detect_class_metrics(all_m)
        chart_data = {
            "run_history": results,
            "metric_keys": chart_metrics,
            "target": final_summary.get("primary_metric", {}).get("target", 0),
            "primary_name": primary_name,
            "class_cats": class_cats,
        }

    primary_name_4 = (final_summary.get("primary_metric", {}).get("name", "accuracy")
                      if has_results else "")
    sections.append((f"{sec_num}. Hypothesis Verification & Conclusion",
                     _build_section4(results or [], final_summary or {},
                                     primary_name_4, hyp, val_n, has_results)))
    sec_num += 1

    if papers_list:
        sections.append((f"{sec_num}. References",
                         _build_section5(papers_list)))

    # 렌더링
    topic_label = inp.get("topic", "Research")
    total_pages = _render_decision_memo(
        cv, W, H, FN, FB, f"Research Decision Memo — {topic_label}",
        sections, chart_data)

    cv.save()
    print(f"  PDF 저장: {output_path}  ({total_pages}페이지, max {MAX_PAGES})")


# ──────────────────────────────────────────────────────────
# 승인 요청
# ──────────────────────────────────────────────────────────

def request_approval(
    topic_file: str,
    hypothesis_file: str,
    validation_file: str,
    papers_file: str = "",
    results_file: str = "",
    summary_file: str = "",
) -> dict:
    topic      = _load_json(topic_file)
    hypothesis = _load_json(hypothesis_file)
    validation = _load_json(validation_file)
    papers     = _load_json(papers_file) if papers_file else {}

    # 실험 결과 로드 (선택)
    results = None
    final_summary = None
    if results_file and Path(results_file).exists():
        lines = Path(results_file).read_text(encoding="utf-8").strip().split("\n")
        results = [json.loads(l) for l in lines if l.strip()]
    if summary_file and Path(summary_file).exists():
        final_summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))

    # validation에 최고 점수 가설이 있으면 교체
    _PACKAGE_FIELDS = ("research_gap", "experiment_plan", "related_papers",
                       "risk_factors", "evidence_pack", "falsification_criteria",
                       "evidence_links")
    if "hypothesis" in validation:
        val_hyp = validation["hypothesis"]
        if isinstance(val_hyp, dict) and "hypothesis" in val_hyp:
            hypothesis["hypothesis"] = val_hyp["hypothesis"]
            for field in _PACKAGE_FIELDS:
                if field in val_hyp:
                    hypothesis[field] = val_hyp[field]
        else:
            hypothesis["hypothesis"] = val_hyp

    topic_name = topic.get("input", {}).get("topic", "research")
    topic_slug = _topic_slug(topic_name)
    reports_dir = _reports_dir(topic_slug)
    summary    = validation.get("summary", {})
    avg_score  = summary.get("average_score", 0)
    below_thr  = avg_score < PDF_SCORE_THRESHOLD

    # 항상 PDF 생성
    pdf_path = reports_dir / "report.pdf"
    print("\n  PDF 보고서 생성 중...")
    generate_pdf(topic, hypothesis, papers, validation, pdf_path,
                 results=results, final_summary=final_summary)

    hyp = hypothesis.get("hypothesis", hypothesis)
    print("\n" + "=" * 60)
    if below_thr:
        print(f"  [!] 점수 미달: {avg_score}/10  (목표 {PDF_SCORE_THRESHOLD}점)")
        print(f"      자동 개선 후 최고 점수 가설로 보고합니다.")
        print(f"  약점: {' / '.join(summary.get('all_weaknesses', [])[:2])}")
    else:
        print(f"  LLM 판정: {summary.get('overall_verdict', '')} ({avg_score}/10)")
    print(f"  가설: {hyp.get('statement_kr', hyp.get('statement', ''))[:80]}")
    print(f"  PDF: {pdf_path.resolve()}")
    print("=" * 60)
    print("\n  [approve] 승인 → 코드 분석·모델 생성 진행")
    print("  [revise]  수정 의견 입력 → 파이프라인 종료")
    print("  [reject]  거부 → 파이프라인 종료")

    while True:
        raw = input("\n결정 (approve/revise/reject): ").strip().lower()
        if raw in ("approve", "revise", "reject"):
            break
        print("  approve / revise / reject 중 하나를 입력하세요.")

    comment = ""
    if raw in ("revise", "reject"):
        comment = input("이유 (선택, Enter 건너뜀): ").strip()

    result = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic_name,
        "decision": raw,
        "comment": comment,
        "pdf_report": str(pdf_path),
        "hypothesis_file": hypothesis_file,
        "validation_summary": summary,
    }
    ap_path = reports_dir / "approval.json"
    ap_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  결정 저장: {ap_path}  |  결정: {raw.upper()}")
    return result


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="사용자 승인 + PDF 보고서")
    parser.add_argument("--topic-file",      required=True)
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--papers-file",     default="")
    parser.add_argument("--results-file",    default="",
                        help="previous_results.jsonl 경로 (실험 결과 포함 시)")
    parser.add_argument("--summary-file",    default="",
                        help="result_summary.json 경로 (실험 결과 포함 시)")
    args = parser.parse_args()

    result = request_approval(
        args.topic_file,
        args.hypothesis_file,
        args.validation_file,
        args.papers_file,
        args.results_file,
        args.summary_file,
    )
    print(f"\n최종 결정: {result['decision']}")
