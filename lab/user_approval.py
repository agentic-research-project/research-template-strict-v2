"""
Stage 5: 사용자 승인 인터페이스

전 단계 결과를 PDF 보고서로 생성하고 사용자의 승인/수정/거부를 받는다.
- PDF: experiments/{slug}/reports/report.pdf  (A4 최대 4페이지, reportlab)
- JSON: experiments/{slug}/reports/approval.json

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
    }


def _norm_validation(v: dict) -> dict:
    """validation.json을 PDF 렌더링용 통합 구조로 변환."""
    # gpt4o/gemini 키 우선, evaluator_1/2 fallback
    eval1 = v.get("gpt4o") or v.get("evaluator_1", {})
    eval2 = v.get("gemini") or v.get("evaluator_2", {})

    # 평가자 이름 추출
    name1 = eval1.get("evaluator", "OpenAI")
    name2 = eval2.get("evaluator", "Gemini")
    # 축약 표시
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
        # 단어 하나가 max_w보다 길면 글자 단위로 분할
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
            # 남은 chunk를 cur로 이어감
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
    # max_lines는 호환성을 위해 유지하되 생략 표시(...)는 하지 않음
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


def _arrow(c, x1: float, y: float, x2: float, color: str = C_GRAY):
    c.setStrokeColor(_color(color))
    c.setLineWidth(1.2)
    c.line(x1, y, x2 - 3, y)
    c.setFillColor(_color(color))
    p = c.beginPath()
    p.moveTo(x2, y)
    p.lineTo(x2 - 4, y + 2.5)
    p.lineTo(x2 - 4, y - 2.5)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def _chip(c, x: float, y: float, w: float, h: float,
          bg: str, text: str, font: str, size: float, color: str):
    _rect(c, x, y, w, h, bg, radius=2)
    c.setFont(font, size)
    c.setFillColor(_color(color))
    c.drawString(x + 2, y + (h - size) / 2 + 1, text)


def _find_paper(papers_list: list, hint) -> dict:
    h = (hint.get("title", "") if isinstance(hint, dict) else str(hint)).lower()
    for p in papers_list:
        if any(w in p.get("title", "").lower() for w in h.split()[:3]):
            return p
    return None


def _draw_bar(c, x: float, by: float, bar_w: float, bar_h: float,
              score: float, verdict: str, label: str,
              bar_color_map: dict, font_reg: str, font_bold: str, lbl_size: float):
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    bc = bar_color_map.get(verdict, C_GRAY)
    c.setFont(font_reg, lbl_size)
    c.setFillColor(HexColor("#333333"))
    c.drawString(x, by + bar_h + 1.2 * mm, label)
    _rect(c, x, by, bar_w, bar_h, "#D5D8DC")
    if score > 0:
        _rect(c, x, by, bar_w * score / 10.0, bar_h, bc)
    c.setFont(font_bold, lbl_size)
    c.setFillColor(HexColor("#222222"))
    c.drawString(x + bar_w + 2 * mm, by + bar_h / 2 - 2 * mm,
                 f"{score}  [{verdict}]")


def _draw_stage_box(c, bx: float, box_bot: float, box_w: float, box_h: float,
                    label: str, color: str, lbl_line_h: float,
                    font_bold: str, lbl_size: float):
    from reportlab.lib.colors import white
    _rect(c, bx, box_bot, box_w, box_h, color, radius=3)
    lines = label.split("\n")
    ly = box_bot + box_h - lbl_line_h
    for ln in lines:
        c.setFont(font_bold, lbl_size)
        c.setFillColor(white)
        lw = _str_width(c, ln, font_bold, lbl_size)
        c.drawString(bx + (box_w - lw) / 2, ly, ln)
        ly -= lbl_line_h


def _draw_bullet_list(c, items: list, x: float, y: float, w: float,
                      font: str, size: float, color: str,
                      item_lh: float, item_gap: float) -> float:
    ry = y
    for item in items:
        ry = _text(c, f"• {item}", x, ry, w, font, size, color, line_h=item_lh)
        ry -= item_gap
    return ry


def _draw_page_header(c, W, H, FN, FB, title: str, subtitle: str = "",
                      page_num: int = 0, total_pages: int = 0):
    """각 페이지 상단 헤더를 그린다."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import white, HexColor
    HDR_H = 14 * mm
    _rect(c, 0, H - HDR_H, W, HDR_H, C_BG)
    c.setFillColor(white)
    c.setFont(FB, 16)
    c.drawCentredString(W / 2, H - HDR_H + HDR_H * 0.60, title)
    c.setFont(FN, 8)
    c.setFillColor(HexColor(C_GRAY))
    sub = subtitle or datetime.now().strftime("%Y-%m-%d")
    page_str = f"   |   {page_num}/{total_pages}" if total_pages else ""
    c.drawCentredString(W / 2, H - HDR_H + HDR_H * 0.22, sub + page_str)
    return H - HDR_H - 3 * mm


# ──────────────────────────────────────────────────────────
# Page 1+: 가설 보고서 (flowing layout — 내용에 따라 자동 확장)
# ──────────────────────────────────────────────────────────

def _page_break_if_needed(c, cur_y, needed_h, W, H, FN, FB, title, subtitle, page_num, total_pages, ML):
    """공간 부족 시 새 페이지를 시작하고 새 cur_y를 반환한다."""
    from reportlab.lib.units import mm
    BOTTOM = 12 * mm
    if cur_y - needed_h < BOTTOM:
        c.showPage()
        page_num += 1
        cur_y = _draw_page_header(c, W, H, FN, FB, title, subtitle, page_num, total_pages)
        cur_y -= 3 * mm
    return cur_y, page_num


def _draw_page_hypothesis(c, W, H, FN, FB, inp, hyp, val_norm, papers_list, below, page_num, total_pages):
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white

    ML = 8 * mm
    CW = W - 2 * ML
    BOTTOM = 12 * mm
    topic_label = inp.get("topic", "Research")
    hdr_sub = f"주제: {topic_label}   |   {datetime.now().strftime('%Y-%m-%d')}"
    cur_y = _draw_page_header(c, W, H, FN, FB, "Research Hypothesis Report",
                              hdr_sub, page_num, total_pages)

    summary  = val_norm["summary"]
    eval1    = val_norm["eval1"]
    eval2    = val_norm["eval2"]
    name1    = val_norm["name1"]
    name2    = val_norm["name2"]
    avg      = summary.get("average_score", 0)
    gap      = hyp["research_gap"]
    exp_plan = hyp["experiment_plan"]
    risks    = hyp["risk_factors"]

    CHIP_H      = 5 * mm
    CHIP_GAP    = 4 * mm
    PAD         = 3 * mm
    CHIP_OFFSET = 0.5 * mm

    # ═══ 1. HYPOTHESIS + LLM SCORE ═══
    HYP_W = CW * 0.65
    LLM_X = ML + HYP_W + CW * 0.03
    LLM_W = CW - HYP_W - CW * 0.03

    stmt_kr = hyp["statement_kr"]
    innov   = hyp["key_innovation"]

    # 실제 콘텐츠 높이 계산 (고정 예산 대신)
    stmt_h  = _text_height(c, stmt_kr, HYP_W - 8*mm, FB, 7.5, line_h=11.5)
    innov_h = _text_height(c, "핵심 혁신: " + innov, HYP_W - 8*mm, FN, 6.5, line_h=8.5)
    INNOV_H = innov_h + 2 * PAD

    bar_h      = 6 * mm
    bar_gap_v  = 6.5 * mm
    llm_bars_h = 3 * (bar_h + bar_gap_v)

    # HYP_H: 가설 텍스트 + 혁신 박스 vs LLM 바 높이 중 큰 값
    hyp_content_h = CHIP_H + CHIP_GAP + stmt_h + PAD + INNOV_H + PAD
    llm_content_h = CHIP_H + CHIP_GAP + llm_bars_h + PAD
    HYP_H = max(hyp_content_h, llm_content_h)

    cur_y, page_num = _page_break_if_needed(
        c, cur_y, HYP_H, W, H, FN, FB, "Research Hypothesis Report", hdr_sub,
        page_num, total_pages, ML)

    hyp_top = cur_y
    hyp_bot = cur_y - HYP_H

    _rect(c, ML, hyp_bot, HYP_W, HYP_H, C_HYPO, radius=3)
    chip_bottom = hyp_top - CHIP_H - CHIP_OFFSET
    _chip(c, ML + 2 * mm, chip_bottom, 50 * mm, CHIP_H,
          C_LIGHT, "핵심 가설 (Core Hypothesis)", FB, 6.5, C_HYPO)

    text_top = chip_bottom - CHIP_GAP
    stmt_end = _text(c, stmt_kr, ML + 4*mm, text_top, HYP_W - 8*mm,
                     FB, 7.5, "white", line_h=11.5)

    innov_top = stmt_end - PAD
    innov_bot = innov_top - INNOV_H
    _rect(c, ML + 2*mm, innov_bot, HYP_W - 4*mm, INNOV_H, "#1A1A2E", radius=2)
    _text(c, "핵심 혁신: " + innov,
          ML + 4*mm, innov_bot + INNOV_H - PAD, HYP_W - 8*mm,
          FN, 6.5, C_GOLD, line_h=8.5)

    # LLM 검증 바
    llm_chip_bottom = hyp_top - CHIP_H - CHIP_OFFSET
    _chip(c, LLM_X, llm_chip_bottom, 22 * mm, CHIP_H,
          C_LIGHT, "LLM 검증", FB, 6.5, C_HYPO)

    bar_color_map = {"approve": C_APPROVE, "revise": C_WARN, "reject": C_ACCENT}
    bar_max_w = LLM_W - 16 * mm
    lbl_size  = 6.5
    bar_start = llm_chip_bottom - CHIP_GAP - lbl_size - bar_h

    bar_data = [
        (name1,  eval1.get("score", 0), eval1.get("verdict", "")),
        (name2,  eval2.get("score", 0), eval2.get("verdict", "")),
        ("종합", avg,                   summary.get("overall_verdict", "")),
    ]
    for i, (lbl, score, verdict) in enumerate(bar_data):
        by = bar_start - i * (bar_h + bar_gap_v)
        _draw_bar(c, LLM_X, by, bar_max_w, bar_h,
                  score, verdict, lbl, bar_color_map, FN, FB, lbl_size)

    c.setFont(FN, 6)
    c.setFillColor(HexColor(C_GRAY))
    c.drawCentredString(LLM_X + bar_max_w / 2, hyp_bot + 3 * mm, "점수 / 10")

    cur_y = hyp_bot - 3 * mm

    # ═══ 2. RESEARCH PIPELINE ═══
    box_h       = 16 * mm
    pipe_hdr    = 8 * mm
    sub_gap_v   = 2 * mm
    sub_lh      = 7
    _pipe_box_w = CW * 0.185

    lims    = gap.get("limitations", [])
    gap_sum = gap.get("summary", "")
    metric  = ", ".join(inp.get("target_metric", "Accuracy").split(",")[:2]).strip()

    stages = [
        ("기존 방법\n(Existing)",  C_GRAY,    lims[0] if lims else ""),
        ("연구 갭\n(Gap)",          C_ACCENT,  gap_sum),
        ("제안 방법\n(Proposed)",   C_HYPO,    hyp["statement_kr"]),
        ("예상 성과\n(Outcome)",    C_APPROVE, metric),
    ]
    # 서브텍스트 최대 높이 계산
    box_w   = _pipe_box_w
    sub_heights = [_text_height(c, sub, box_w - 2*mm, FN, 5.5, line_h=sub_lh)
                   for _, _, sub in stages]
    max_sub_h = max(sub_heights) if sub_heights else 0
    PIPE_H = pipe_hdr + box_h + sub_gap_v + max_sub_h + 3 * mm

    cur_y, page_num = _page_break_if_needed(
        c, cur_y, PIPE_H, W, H, FN, FB, "Research Hypothesis Report", hdr_sub,
        page_num, total_pages, ML)

    pipe_top = cur_y
    pipe_bot = cur_y - PIPE_H

    _rect(c, ML, pipe_bot, CW, PIPE_H, "#F8F9FA", radius=3)
    _chip(c, ML + 2*mm, pipe_top - 5.5*mm, 38*mm, 5*mm,
          C_LIGHT, "Research Pipeline", FB, 6.5, C_HYPO)

    box_spc    = (CW - 4 * box_w) / 3
    lbl_line_h = 4.5 * mm
    box_top_y  = pipe_top - pipe_hdr
    box_bot_y  = box_top_y - box_h
    sub_top    = box_bot_y - sub_gap_v

    for i, (label, color, sub) in enumerate(stages):
        bx = ML + i * (box_w + box_spc)
        _draw_stage_box(c, bx, box_bot_y, box_w, box_h, label, color, lbl_line_h, FB, 7)
        if i < 3:
            ax1 = bx + box_w + 1 * mm
            ax2 = bx + box_w + box_spc - 1 * mm
            _arrow(c, ax1, box_bot_y + box_h / 2, ax2, color=C_GRAY)
        _text(c, sub, bx + 1*mm, sub_top, box_w - 2*mm,
              FN, 5.5, color, line_h=sub_lh, align="center")

    cur_y = pipe_bot - 3 * mm

    # ═══ 3. NOVELTY (key_innovation / mechanism) ═══
    key_innov = hyp["key_innovation"]
    mechanism = hyp["expected_mechanism"]

    nov_gap_w = CW * 0.03
    nov_w     = (CW - nov_gap_w) / 2
    nov_lh    = 8.5

    # 실제 텍스트 높이로 섹션 높이 결정
    ki_h = _text_height(c, key_innov, nov_w - 6*mm, FN, 6.5, line_h=nov_lh)
    me_h = _text_height(c, mechanism, nov_w - 6*mm, FN, 6.5, line_h=nov_lh)
    NOV_H = CHIP_H + CHIP_GAP + max(ki_h, me_h) + PAD

    cur_y, page_num = _page_break_if_needed(
        c, cur_y, NOV_H, W, H, FN, FB, "Research Hypothesis Report", hdr_sub,
        page_num, total_pages, ML)

    nov_top = cur_y
    nov_bot = cur_y - NOV_H
    nov_rx  = ML + nov_w + nov_gap_w

    _rect(c, ML,     nov_bot, nov_w, NOV_H, "#EEF4FF", radius=3)
    _rect(c, nov_rx, nov_bot, nov_w, NOV_H, "#F0FFF4", radius=3)

    _chip(c, ML     + 2*mm, nov_top - CHIP_H - CHIP_OFFSET, 44*mm, CHIP_H,
          C_HYPO,    "핵심 혁신 (Key Innovation)", FB, 6.5, "white")
    _chip(c, nov_rx + 2*mm, nov_top - CHIP_H - CHIP_OFFSET, 38*mm, CHIP_H,
          C_APPROVE, "작동 원리 (Mechanism)",      FB, 6.5, "white")

    text_start = nov_top - CHIP_H - CHIP_GAP
    _text(c, key_innov, ML     + 3*mm, text_start, nov_w - 6*mm,
          FN, 6.5, "#1A3A6B", line_h=nov_lh)
    _text(c, mechanism, nov_rx + 3*mm, text_start, nov_w - 6*mm,
          FN, 6.5, "#1A5C3A", line_h=nov_lh)

    cur_y = nov_bot - 3 * mm

    # ═══ 4. 실험 계획 (왼쪽 63%) + 근거 & 리스크 (오른쪽 34%) ═══
    EXP_W  = CW * 0.63
    RISK_X = ML + EXP_W + CW * 0.03
    RISK_W = CW - EXP_W - CW * 0.03

    baselines    = exp_plan.get("baseline_models", [])
    metrics_list = exp_plan.get("evaluation_metrics", [])
    exp_rows = [
        ("아키텍처", exp_plan.get("architecture", "")),
        ("데이터셋",  exp_plan.get("dataset", "")),
        ("기준 모델", "  /  ".join(str(b).split("(")[0].strip() for b in baselines[:4])),
        ("평가 지표", "  /  ".join(str(m).split("—")[0].strip() for m in metrics_list[:4])),
    ]
    lbl_col_w = 18 * mm
    lbl_x     = ML + PAD
    val_x     = lbl_x + lbl_col_w
    val_w     = EXP_W - lbl_col_w - 2 * PAD
    row_gap   = 2 * mm
    exp_hdr_h = 9 * mm
    exp_size  = 8.0
    row_lh    = exp_size * 1.45

    # 실험계획 실제 높이
    exp_content_h = exp_hdr_h + sum(
        _text_height(c, val, val_w, FN, exp_size, line_h=row_lh) + row_gap
        for _, val in exp_rows
    ) + PAD

    # 근거&리스크 실제 높이
    strengths   = summary.get("all_strengths", [])
    item_size   = 7.0
    item_lh_v   = item_size * 1.45
    item_gap    = 1.5 * mm
    lbl_gap     = 5 * mm
    section_gap = 5 * mm
    risk_hdr_h  = 10 * mm

    strengths_h = sum(
        _text_height(c, f"• {s}", RISK_W - 6*mm, FN, item_size, line_h=item_lh_v) + item_gap
        for s in strengths[:3]
    )
    risks_h = sum(
        _text_height(c, f"• {r}", RISK_W - 6*mm, FN, item_size, line_h=item_lh_v) + item_gap
        for r in risks[:3]
    )
    risk_content_h = risk_hdr_h + lbl_gap + strengths_h + section_gap + lbl_gap + risks_h + PAD

    SECTION_H = max(exp_content_h, risk_content_h)

    cur_y, page_num = _page_break_if_needed(
        c, cur_y, SECTION_H, W, H, FN, FB, "Research Hypothesis Report", hdr_sub,
        page_num, total_pages, ML)

    exp_top = cur_y
    exp_bot = cur_y - SECTION_H

    _rect(c, ML, exp_bot, EXP_W, SECTION_H, C_LIGHT, radius=3)
    _chip(c, ML + 2*mm, exp_top - 5.5*mm, 54*mm, 5*mm,
          C_HYPO, "실험 계획 (Experiment Plan)", FB, 6.5, "white")

    row_y = exp_top - exp_hdr_h
    for lbl, val in exp_rows:
        c.setFont(FB, 7)
        c.setFillColor(HexColor(C_HYPO))
        c.drawString(lbl_x, row_y, f"[{lbl}]")
        next_y = _text(c, val, val_x, row_y, val_w,
                       FN, exp_size, "#333333", line_h=row_lh)
        row_y = next_y - row_gap

    # 근거 & 리스크
    _rect(c, RISK_X, exp_bot, RISK_W, SECTION_H, "#FFF8EE", radius=3)
    _chip(c, RISK_X + 2*mm, exp_top - 5.5*mm, 26*mm, 5*mm,
          C_HYPO, "근거 & 리스크", FB, 6.5, "white")

    c.setFont(FB, 7)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(RISK_X + 3*mm, exp_top - risk_hdr_h, "근거")
    ry = _draw_bullet_list(c, strengths[:3],
                           RISK_X + 3*mm, exp_top - risk_hdr_h - lbl_gap,
                           RISK_W - 6*mm, FN, item_size, "#222222", item_lh_v, item_gap)

    c.setFont(FB, 7)
    c.setFillColor(HexColor(C_ACCENT))
    c.drawString(RISK_X + 3*mm, ry - section_gap, "리스크")
    _draw_bullet_list(c, risks[:3],
                      RISK_X + 3*mm, ry - section_gap - lbl_gap,
                      RISK_W - 6*mm, FN, item_size, "#555555", item_lh_v, item_gap)

    cur_y = exp_bot - PAD

    # ═══ 5. VERDICT ═══
    vrd_main_size = 8
    vrd_sub_size  = 6
    overall = summary.get("overall_verdict", "")
    if below:
        vrd_main_text = f"[!] 점수 미달  {avg}/10  (목표 {PDF_SCORE_THRESHOLD}점) — 사용자 판단 필요"
        vrd_sub_text  = f"주요 약점: {summary.get('all_weaknesses', ['약점 정보 없음'])[0]}"
    else:
        v_label = {"approve": "[OK] 승인 권고",
                   "revise":  "[!] 수정 권고",
                   "reject":  "[X] 거부 권고"}.get(overall, overall)
        vrd_main_text = f"LLM 종합 판정:  {v_label}   (평균 {avg} / 10)"
        vrd_sub_text  = "AI-Driven Research Automation  |  Generated by Claude"

    vrd_main_h = _text_height(c, vrd_main_text, CW - 4*PAD, FB, vrd_main_size)
    vrd_sub_h  = _text_height(c, vrd_sub_text, CW - 4*PAD, FN, vrd_sub_size)
    FTR_H = vrd_main_h + vrd_sub_h + 4 * PAD

    cur_y, page_num = _page_break_if_needed(
        c, cur_y, FTR_H, W, H, FN, FB, "Research Hypothesis Report", hdr_sub,
        page_num, total_pages, ML)

    vrd_bot = cur_y - FTR_H
    vrd_top = cur_y

    if below:
        _rect(c, ML, vrd_bot, CW, FTR_H, "#7D3C00", radius=3)
        y = _text(c, vrd_main_text,
                  ML + 2*PAD, vrd_top - PAD, CW - 4*PAD,
                  FB, vrd_main_size, "#FFD580", align="center")
        _text(c, vrd_sub_text,
              ML + 2*PAD, y - PAD, CW - 4*PAD,
              FN, vrd_sub_size, "#FFECB3", align="center")
    else:
        v_color = {"approve": C_APPROVE, "revise": C_WARN,
                   "reject": C_ACCENT}.get(overall, "white")
        _rect(c, ML, vrd_bot, CW, FTR_H, C_BG, radius=3)
        y = _text(c, vrd_main_text,
                  ML + 2*PAD, vrd_top - PAD, CW - 4*PAD,
                  FB, vrd_main_size, v_color, align="center")
        _text(c, vrd_sub_text,
              ML + 2*PAD, y - PAD, CW - 4*PAD,
              FN, vrd_sub_size, C_GRAY, align="center")


# ──────────────────────────────────────────────────────────
# Page 2: 실험 결과
# ──────────────────────────────────────────────────────────

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
#
# 공통 구조:
#   _chart_setup()   — 배경, 축, 눈금, target 라인, 범위 계산
#   _chart_legend()  — 범례
#   _chart_bar()     — bar / grouped bar
#   _chart_line()    — line / multi-line
#   _chart_hbar()    — horizontal bar (카테고리별 비교)
#   _draw_chart()    — 자동 선택 dispatcher

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
    """all_metrics에서 클래스별 metric을 추출한다.
    예: class_0_acc, class_1_acc, ... → {"0": 0.908, "1": 0.989, ...}
    """
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
    """차트 공통 셋업: 배경, 제목, y축, target 라인.
    Returns (chart_x, chart_y, chart_w, chart_h, v_min, v_max, v_range).
    """
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    _rect(c, x, y, w, h, "#F8F9FA", radius=4)

    PAD = 4 * mm
    Y_LABEL_W = 12 * mm
    chart_x = x + PAD + Y_LABEL_W
    chart_y = y + PAD + 6 * mm
    chart_w = w - 2 * PAD - Y_LABEL_W - 5 * mm
    chart_h = h - 2 * PAD - 14 * mm

    # 제목
    c.setFont(FB, 9)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(x + PAD, y + h - PAD - 3 * mm, title)

    # 범위
    if target and isinstance(target, (int, float)):
        all_vals = list(all_vals) + [target]
    v_min_raw = min(all_vals) if all_vals else 0
    v_max_raw = max(all_vals) if all_vals else 1
    margin = (v_max_raw - v_min_raw) * 0.1 or 0.005
    v_min = v_min_raw - margin
    v_max = v_max_raw + margin
    v_range = v_max - v_min if v_max > v_min else 0.01

    # y축 눈금
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

    # target 라인
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

        # 선 그리기
        if len(points) >= 2:
            p = c.beginPath()
            p.moveTo(points[0][0], points[0][1])
            for px, py, _ in points[1:]:
                p.lineTo(px, py)
            c.drawPath(p, fill=0, stroke=1)

        # 점 + 값 라벨
        for pi, (px, py, val) in enumerate(points):
            c.setFillColor(HexColor(color))
            c.circle(px, py, 2.5, fill=1, stroke=0)

            if len(metric_keys) <= 2:
                fmt = _auto_format(val, max(v[2] for v in points))
                c.setFont(FB, 5.5)
                c.setFillColor(HexColor("#333333"))
                c.drawString(px + 2 * mm, py + 1.5 * mm, fmt)

    # x축 라벨
    for vi, entry in enumerate(run_history):
        px = cx + (vi / max(n - 1, 1)) * cw
        lbl = f"v{entry.get('version', vi+1)}"
        c.setFont(FN, 7)
        c.setFillColor(HexColor("#555555"))
        lw = _str_width(c, lbl, FN, 7)
        c.drawString(px - lw / 2, cy - 4 * mm, lbl)


def _chart_hbar(c, x, y, w, h, categories: dict[str, float],
                target, title, FN, FB):
    """Horizontal bar chart — 카테고리별 비교 (클래스별 정확도 등)."""
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

    # target 세로 라인
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

        # 라벨
        c.setFont(FN, 6)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(chart_x - 2 * mm, by + bar_h / 2 - 2, label)

        # 바
        _rect(c, chart_x, by, max(bw, 1), bar_h, color, radius=2)

        # 값
        fmt = _auto_format(val, v_max)
        c.setFont(FB, 5.5)
        c.setFillColor(HexColor("#333333"))
        c.drawString(chart_x + bw + 1.5 * mm, by + bar_h / 2 - 2, fmt)


def _draw_chart(c, x, y, w, h, run_history, metric_keys, target,
                primary_name, FN, FB):
    """데이터 형태에 따라 차트 타입을 자동 선택하여 그린다.

    선택 기준:
      버전 ≤ 5, metric 1개  → bar chart
      버전 ≤ 5, metric 2개+ → grouped bar chart
      버전 > 5, metric 1개  → line chart
      버전 > 5, metric 2개+ → multi-line chart
    """
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

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

    # 제목
    if n_metrics == 1:
        title = f"{primary_name.replace('_', ' ').title()} Progression"
    else:
        title = "Metrics Comparison"

    # 범례 공간 예약
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


# ── 분석 텍스트 생성 ──────────────────────────────────────

def _paper_ref(evidence_links: list, related_papers: list, aspect: str) -> str:
    """evidence_links에서 aspect를 지원하는 논문 번호를 찾아 참조 문자열을 반환한다.
    예: "[1][4]" 또는 빈 문자열.
    """
    refs = []
    for link in evidence_links:
        supports = link.get("supports", [])
        if aspect in supports or any(aspect in s for s in supports):
            refs.append(link.get("paper_id", ""))
    if not refs:
        # related_papers에서 aspect 키워드로 fallback 탐색
        for rp in related_papers:
            rel = rp.get("relevance", "").lower()
            if aspect.lower().replace("_", " ") in rel:
                refs.append(rp.get("paper_id", ""))
    if refs:
        return " " + "".join(f"[{r}]" for r in refs[:3])
    return ""


def _build_analysis(run_history: list, final_summary: dict, primary_name: str,
                    hypothesis: dict = None) -> list[str]:
    """가설에서 결과까지의 전체 과정을 근거 논문 참조와 함께 상세히 분석한다."""
    lines = []
    if not run_history:
        return lines

    primary = final_summary.get("primary_metric", {})
    all_m   = final_summary.get("all_metrics", {})
    target  = primary.get("target", 0)
    met     = primary.get("met", False)
    n_runs  = len(run_history)
    first_val = run_history[0].get(primary_name, 0)
    last_val  = run_history[-1].get(primary_name, 0)

    hyp = hypothesis or {}
    ev_links = hyp.get("evidence_links", [])
    if not isinstance(ev_links, list):
        ev_links = []
    rel_papers = hyp.get("related_papers", [])
    if not isinstance(rel_papers, list):
        rel_papers = []

    # ── 1) 가설 근거 및 핵심 주장 ──
    rationale = hyp.get("rationale", "")
    innov     = hyp.get("key_innovation", "")
    ref_innov = _paper_ref(ev_links, rel_papers, "key_innovation")
    ref_rat   = _paper_ref(ev_links, rel_papers, "rationale")

    if rationale:
        lines.append(
            f"연구 근거: {rationale}{ref_rat}"
        )

    if innov:
        status_str = "실험을 통해 유효성이 확인되었다" if met else "부분적으로 검증되었으나 목표 달성에는 미달하였다"
        lines.append(
            f"핵심 혁신: {innov}{ref_innov} "
            f"— 본 혁신은 {n_runs}회 반복 실험을 거쳐 {status_str}."
        )

    # ── 2) 예상 메커니즘 vs 실제 결과 ──
    mechanism = hyp.get("expected_mechanism", "")
    ref_mech  = _paper_ref(ev_links, rel_papers, "expected_mechanism")
    if mechanism:
        lines.append(
            f"예상 메커니즘: {mechanism}{ref_mech}"
        )
        if met:
            lines.append(
                f"검증 결과: {primary_name} {first_val:.4f} → {last_val:.4f}로 "
                f"목표 {target:.4f}를 초과 달성(+{last_val - target:.4f})하여, "
                f"제안된 메커니즘이 해당 과제에서 효과적임이 실증되었다."
            )
        else:
            lines.append(
                f"검증 결과: {primary_name} {last_val:.4f}로 목표 {target:.4f} 대비 "
                f"{target - last_val:.4f} 부족. 메커니즘의 기본 방향은 유효하나 "
                f"추가적인 구조 개선이 필요하다."
            )

    # ── 3) 버전별 변경 영향 분석 ──
    ref_arch = _paper_ref(ev_links, rel_papers, "architecture")
    lines.append(
        f"실험 진행: 총 {n_runs}회 반복 실험을 수행하였다.{ref_arch}"
    )
    for i in range(len(run_history)):
        entry = run_history[i]
        val = entry.get(primary_name, 0)
        changes = entry.get("changes", "")
        params  = entry.get("params_M", "?")

        if i == 0:
            lines.append(
                f"  v{entry.get('version','?')} (baseline): {primary_name}={val:.4f}, "
                f"{params}M params. 초기 구조: {changes}."
            )
        else:
            prev_val = run_history[i-1].get(primary_name, 0)
            delta = val - prev_val
            sign = "+" if delta >= 0 else ""
            pct = (delta / prev_val * 100) if prev_val else 0
            if delta > 0:
                effect = f"{sign}{delta:.4f} ({sign}{pct:.2f}%) 개선"
            elif delta < 0:
                effect = f"{delta:.4f} ({pct:.2f}%) 하락"
            else:
                effect = "변화 없음"
            lines.append(
                f"  v{entry.get('version','?')}: {changes} → {effect}. "
                f"{primary_name}={val:.4f}, {params}M params."
            )

    # ── 4) 클래스별 상세 분석 ──
    class_accs = {k: v for k, v in all_m.items()
                  if k.startswith("class_") and k.endswith("_acc")}
    if class_accs:
        best_cls  = max(class_accs, key=class_accs.get)
        worst_cls = min(class_accs, key=class_accs.get)
        best_idx  = best_cls.replace("class_", "").replace("_acc", "")
        worst_idx = worst_cls.replace("class_", "").replace("_acc", "")
        spread = class_accs[best_cls] - class_accs[worst_cls]

        lines.append(
            f"클래스별 성능 분석: 최고 class {best_idx} ({class_accs[best_cls]:.3f}), "
            f"최저 class {worst_idx} ({class_accs[worst_cls]:.3f}). "
            f"클래스 간 편차 {spread:.3f}."
        )
        if spread > 0.15:
            lines.append(
                f"  class {worst_idx}의 정확도가 상대적으로 낮아 "
                f"해당 클래스에 대한 특화 전략(focal loss, 증강 강화 등)이 필요하다."
            )
        else:
            lines.append(
                f"  전체 클래스에 걸쳐 균형 잡힌 분류 성능을 달성하였다."
            )

    # ── 5) 파라미터 효율성 분석 ──
    params_list = [e.get("params_M") for e in run_history if e.get("params_M")]
    if params_list:
        first_p = params_list[0]
        last_p  = params_list[-1]
        eff = last_val / last_p if last_p else 0
        lines.append(
            f"파라미터 효율성: {first_p}M → {last_p}M params. "
            f"{primary_name}/M = {eff:.2f}. "
            + (f"파라미터 증가({last_p - first_p:.2f}M) 대비 "
               f"{primary_name} 개선({last_val - first_val:.4f})으로 "
               f"효율적인 스케일업이 이루어졌다." if last_p > first_p
               else f"동일 파라미터 예산 내에서 성능을 개선하였다.")
        )

    # ── 6) 반증 조건(falsification criteria) 검증 ──
    falsif = hyp.get("falsification_criteria", [])
    if falsif:
        lines.append("반증 조건 검증:")
        for fc in falsif:
            fc_text = fc if isinstance(fc, str) else str(fc)
            if "92.5%" in fc_text or "검증 정확도" in fc_text:
                status = "PASS" if met else "FAIL"
                result = f"{primary_name}={last_val:.4f}" + (" ≥ target" if met else " < target")
            elif "파라미터" in fc_text or "200K" in fc_text:
                last_p = run_history[-1].get("params_M", 0)
                status = "PASS" if last_p <= 0.2 else "FAIL"
                result = f"{last_p}M params" + (" ≤ 0.2M" if last_p <= 0.2 else " > 0.2M")
            elif "80%" in fc_text or "클래스" in fc_text:
                min_acc = all_m.get("min_class_accuracy",
                                    min(class_accs.values()) if class_accs else 0)
                status = "PASS" if min_acc >= 0.80 else "WARN"
                result = f"min_class_acc={min_acc:.3f}" + (" ≥ 0.80" if min_acc >= 0.80 else " < 0.80")
            else:
                status = "N/A"
                result = "자동 검증 불가"
            lines.append(f"  [{status}] {fc_text} → {result}")

    # ── 7) Hypothesis Implementation Audit ──
    hyp_impl = final_summary.get("hypothesis_implementation", {})
    if hyp_impl:
        lines.append("가설 구현 감사 (Hypothesis Implementation Audit):")

        mech = hyp_impl.get("mechanism_audit", {})
        if mech:
            status = "PASS" if mech.get("implemented") else "FAIL"
            risk = mech.get("risk_level", "")
            lines.append(f"  [Mechanism {status}] risk={risk}")
            for ev in mech.get("evidence", [])[:3]:
                lines.append(f"    근거: {ev}")
            for ml in mech.get("missing_links", [])[:3]:
                lines.append(f"    누락: {ml}")
            mapping = mech.get("mechanism_mapping", {})
            if mapping:
                for concept, impl in list(mapping.items())[:4]:
                    lines.append(f"    {concept} → {impl}")

        metr = hyp_impl.get("metric_audit", {})
        if metr:
            status = "PASS" if metr.get("implemented") else "FAIL"
            lines.append(
                f"  [Metric {status}] primary={metr.get('primary_metric_expected', '?')}, "
                f"found={metr.get('primary_metric_found', '?')}, "
                f"stdout_ok={metr.get('stdout_contract_ok', '?')}"
            )
            expected = metr.get("required_keys_expected", [])
            found = metr.get("required_keys_found", [])
            if expected:
                missing_keys = [k for k in expected if k not in found]
                if missing_keys:
                    lines.append(f"    누락 키: {', '.join(missing_keys)}")

        cons = hyp_impl.get("constraints_audit", {})
        if cons:
            status = "PASS" if cons.get("implemented") else "FAIL"
            recognized = cons.get("recognized_constraints", [])
            violations = cons.get("violations", [])
            lines.append(
                f"  [Constraints {status}] 인식={len(recognized)}개, 위반={len(violations)}개"
            )
            for v in violations[:3]:
                lines.append(f"    위반: {v}")

    return lines


def _build_conclusion(run_history: list, final_summary: dict, primary_name: str) -> list[str]:
    """실험 결과의 핵심 결론을 요약한다."""
    lines = []
    if not run_history:
        return lines

    primary = final_summary.get("primary_metric", {})
    all_m   = final_summary.get("all_metrics", {})
    target  = primary.get("target", 0)
    met     = primary.get("met", False)
    n_runs  = len(run_history)
    first_val = run_history[0].get(primary_name, 0)
    last_val  = run_history[-1].get(primary_name, 0)
    total_delta = last_val - first_val

    # 1) 핵심 결론
    if met:
        lines.append(
            f"총 {n_runs}회 실험을 통해 {primary_name} {first_val:.4f} → {last_val:.4f} "
            f"(+{total_delta:.4f})로 목표 {target:.4f}를 달성하였다."
        )
    else:
        lines.append(
            f"총 {n_runs}회 실험 결과 {primary_name} {first_val:.4f} → {last_val:.4f} "
            f"(+{total_delta:.4f}). 목표 {target:.4f} 대비 "
            f"{target - last_val:.4f} 부족."
        )

    # 2) 파라미터 효율성
    params_list = [e.get("params_M") for e in run_history if e.get("params_M")]
    if params_list:
        last_p = params_list[-1]
        lines.append(
            f"최종 모델 크기: {last_p}M params "
            f"({primary_name}/M = {last_val/last_p:.2f})."
        )

    # 3) 최종 델타 요약
    last_ver = run_history[-1].get("version", 1)
    deltas = final_summary.get(f"deltas_vs_v{last_ver - 1}", {})
    if deltas:
        improved = [f"{k}(+{v:.4f})" for k, v in deltas.items()
                    if isinstance(v, (int, float)) and v > 0]
        degraded = [f"{k}({v:.4f})" for k, v in deltas.items()
                    if isinstance(v, (int, float)) and v < 0]
        if improved:
            lines.append(f"최종 버전 개선: {', '.join(improved[:4])}")
        if degraded:
            lines.append(f"최종 버전 하락: {', '.join(degraded[:4])}")

    return lines


def _draw_page_results(c, W, H, FN, FB, inp, hyp_norm, final_summary, run_history,
                       page_num, total_pages):
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white

    ML = 10 * mm
    CW = W - 2 * ML

    cur_y = _draw_page_header(c, W, H, FN, FB, "Experiment Results",
                              f"주제: {inp.get('topic', '')}",
                              page_num, total_pages)
    cur_y -= 5 * mm

    primary = final_summary.get("primary_metric", {})
    met     = primary.get("met", False)
    status  = final_summary.get("status", "unknown")
    primary_name = primary.get("name", "accuracy")

    # ── 결과 큰 박스 ──
    result_color = C_APPROVE if met else C_WARN if status == "success" else C_ACCENT
    BOX_H = 22 * mm
    _rect(c, ML, cur_y - BOX_H, CW, BOX_H, result_color, radius=6)

    result_label = "TARGET ACHIEVED" if met else status.upper()
    c.setFont(FB, 14)
    c.setFillColor(white)
    c.drawString(ML + 5 * mm, cur_y - 9 * mm, result_label)

    metric_text = f"{primary_name}: {primary.get('value', 0):.4f}   (target: {primary.get('target', 0):.2f})"
    c.setFont(FN, 11)
    c.drawString(ML + 5 * mm, cur_y - 17 * mm, metric_text)

    sec = final_summary.get("secondary_metrics", [])
    if sec:
        sec_text = " | ".join(f"{m.get('name','?')}: {m.get('value',0):.4f}" for m in sec[:4])
        c.setFont(FN, 8)
        c.setFillColor(HexColor(C_LIGHT))
        c.drawString(ML + CW - _str_width(c, sec_text, FN, 8) - 5 * mm, cur_y - 9 * mm, sec_text)

    cur_y -= BOX_H + 6 * mm

    # ── 일반화 차트 (추이) ──
    chart_metrics = _detect_chart_metrics(run_history, primary_name)[:4]
    all_m = final_summary.get("all_metrics", {})
    class_cats = _detect_class_metrics(all_m)

    # 클래스별 데이터가 있으면 좌우 분할, 없으면 전체 너비
    if class_cats:
        CHART_W_L = CW * 0.55
        CHART_W_R = CW - CHART_W_L - 3 * mm
        CHART_H = 50 * mm
        _draw_chart(c, ML, cur_y - CHART_H, CHART_W_L, CHART_H,
                    run_history, chart_metrics, primary.get("target", 0), primary_name, FN, FB)
        _chart_hbar(c, ML + CHART_W_L + 3 * mm, cur_y - CHART_H, CHART_W_R, CHART_H,
                    class_cats, primary.get("target", 0),
                    "Per-Class Accuracy", FN, FB)
    else:
        CHART_H = 50 * mm
        _draw_chart(c, ML, cur_y - CHART_H, CW, CHART_H,
                    run_history, chart_metrics, primary.get("target", 0), primary_name, FN, FB)
    cur_y -= CHART_H + 6 * mm

    # ── Run History 테이블 ──
    c.setFont(FB, 9)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(ML, cur_y, "Run History")
    cur_y -= 5 * mm

    columns = [
        ("Ver",          0.00, 0.06),
        (primary_name.title(), 0.06, 0.12),
        ("Params",       0.18, 0.08),
        ("Epochs",       0.26, 0.07),
        ("Changes",      0.33, 0.67),
    ]
    ROW_H = 5.5 * mm

    _rect(c, ML, cur_y - ROW_H, CW, ROW_H, C_CARD)
    for label, start, width in columns:
        cx = ML + CW * start
        c.setFont(FB, 6.5)
        c.setFillColor(white)
        c.drawString(cx + 1.5 * mm, cur_y - ROW_H + 1.5 * mm, label)
    cur_y -= ROW_H

    for idx, entry in enumerate(run_history):
        row_bg = C_LIGHT if idx % 2 == 0 else "white"
        _rect(c, ML, cur_y - ROW_H, CW, ROW_H, row_bg)

        pval = entry.get(primary_name, entry.get("accuracy", 0))
        row_data = [
            f"v{entry.get('version', '?')}",
            f"{pval:.4f}" if isinstance(pval, float) else str(pval),
            f"{entry.get('params_M', '?')}M",
            str(entry.get("epochs", "?")),
            str(entry.get("changes", "")),
        ]
        for i, (label, start, width) in enumerate(columns):
            cx = ML + CW * start
            col_w = CW * width - 3 * mm
            val = _truncate(c, row_data[i], FN, 6.5, col_w)
            c.setFont(FN, 6.5)
            c.setFillColor(HexColor("#222222"))
            c.drawString(cx + 1.5 * mm, cur_y - ROW_H + 1.5 * mm, val)

        cur_y -= ROW_H

    cur_y -= 6 * mm

    # ── Analysis: 가설 → 결과 연결 분석 (생략 없이 전량 출력) ──
    analysis = _build_analysis(run_history, final_summary, primary_name,
                               hypothesis=hyp_norm)
    BOTTOM = 15 * mm
    LINE_H = 8.5
    LINE_GAP = 1.5 * mm
    FONT_SZ = 6.5

    if analysis:
        c.setFont(FB, 9)
        c.setFillColor(HexColor(C_HYPO))
        c.drawString(ML, cur_y, "Analysis")
        cur_y -= 5 * mm

        for line in analysis:
            line_h_needed = _text_height(c, f"• {line}", CW - 6 * mm, FN, FONT_SZ, line_h=LINE_H)
            # 페이지 넘김
            if cur_y - line_h_needed - LINE_GAP < BOTTOM:
                c.showPage()
                cur_y = _draw_page_header(c, W, H, FN, FB, "Experiment Results",
                                          f"Analysis (cont.)", page_num, total_pages)
                cur_y -= 5 * mm
            cur_y = _text(c, f"• {line}", ML + 3 * mm, cur_y, CW - 6 * mm,
                          FN, FONT_SZ, "#333333", line_h=LINE_H)
            cur_y -= LINE_GAP

        cur_y -= 3 * mm

    # ── Conclusion ──
    conclusion = _build_conclusion(run_history, final_summary, primary_name)
    if conclusion:
        if cur_y < 40 * mm:
            c.showPage()
            cur_y = _draw_page_header(c, W, H, FN, FB, "Experiment Results",
                                      f"Conclusion", page_num, total_pages)
            cur_y -= 5 * mm

        c.setFont(FB, 9)
        c.setFillColor(HexColor(C_HYPO))
        c.drawString(ML, cur_y, "Conclusion")
        cur_y -= 5 * mm

        concl_h = sum(
            _text_height(c, f"• {line}", CW - 6 * mm, FN, 7, line_h=9) + 1.5 * mm
            for line in conclusion
        )
        box_h = min(concl_h + 4 * mm, cur_y - BOTTOM)
        met_flag = final_summary.get("primary_metric", {}).get("met", False)
        box_color = "#E8F5E9" if met_flag else "#FFF3E0"
        _rect(c, ML, cur_y - box_h, CW, box_h, box_color, radius=4)
        cur_y -= 2 * mm

        for line in conclusion:
            cur_y = _text(c, f"• {line}", ML + 3 * mm, cur_y, CW - 6 * mm,
                          FN, 7, "#333333", line_h=9)
            cur_y -= 1.5 * mm

    # ── Training Stability ──
    stab = final_summary.get("training_stability", {})
    if stab and cur_y > 20 * mm:
        conv = stab.get("loss_converged", False)
        nan_det = stab.get("nan_detected", False)
        stab_color = C_APPROVE if conv and not nan_det else C_WARN
        stab_text = f"Stability — Loss: {'Converged' if conv else 'Not converged'}  |  NaN: {'Detected' if nan_det else 'Clean'}"
        c.setFont(FN, 6.5)
        c.setFillColor(HexColor(stab_color))
        c.drawString(ML + 3 * mm, cur_y, stab_text)

    # 푸터
    _rect(c, 0, 0, W, 8 * mm, C_BG)
    run_id = final_summary.get("run_id", "")
    c.setFont(FN, 6)
    c.setFillColor(HexColor(C_GRAY))
    c.drawString(ML, 2.5 * mm, f"Run: {run_id}")
    c.drawString(W - ML - 40 * mm, 2.5 * mm, datetime.now().strftime("%Y-%m-%d %H:%M"))


# ──────────────────────────────────────────────────────────
# Page 3-4: 참고 문헌 (논문 형식)
# ──────────────────────────────────────────────────────────

def _format_reference(paper: dict, idx: int) -> str:
    """논문을 학술 참고문헌 형식으로 포맷한다.
    [N] Author1, Author2, et al. "Title." Venue, Year.
    """
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


def _draw_pages_references(c, W, H, FN, FB, papers_list, start_page, total_pages):
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    ML = 10 * mm
    CW = W - 2 * ML
    BOTTOM_MARGIN = 15 * mm
    REF_LH = 9.5       # 줄 높이 (pt)
    REF_GAP = 3 * mm    # 항목 간 간격
    REF_SIZE = 7        # 폰트 크기
    page_num = start_page

    cur_y = _draw_page_header(c, W, H, FN, FB, "References",
                              f"참고 논문 {len(papers_list)}편",
                              page_num, total_pages)
    cur_y -= 5 * mm

    for idx, paper in enumerate(papers_list):
        if not paper:
            continue

        ref_text = _format_reference(paper, idx + 1)
        ref_h = _text_height(c, ref_text, CW, FN, REF_SIZE, line_h=REF_LH)

        # 페이지 넘김
        if cur_y - ref_h - REF_GAP < BOTTOM_MARGIN:
            c.showPage()
            page_num += 1
            if page_num > total_pages:
                break
            cur_y = _draw_page_header(c, W, H, FN, FB, "References",
                                      f"참고 논문 (계속)",
                                      page_num, total_pages)
            cur_y -= 5 * mm

        _text(c, ref_text, ML, cur_y, CW, FN, REF_SIZE, "#333333", line_h=REF_LH)
        cur_y -= ref_h + REF_GAP


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
    """reportlab A4 최대 4페이지 PDF 생성.

    Page 1: 가설 보고서
    Page 2: 실험 결과 (results/final_summary 있을 때만)
    Page 3-4: 참고 문헌 (전체 논문)
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4

    FN, FB = _setup_fonts()
    W, H   = A4
    output_path.parent.mkdir(exist_ok=True)
    cv = rl_canvas.Canvas(str(output_path), pagesize=A4)

    # 데이터 정규화
    inp     = topic.get("input", {})
    hyp     = _norm_hypothesis(hypothesis)
    val_n   = _norm_validation(validation)
    summary = val_n["summary"]
    avg     = summary.get("average_score", 0)
    below   = avg < PDF_SCORE_THRESHOLD
    papers_list = papers.get("papers", [])

    has_results = results is not None and final_summary is not None

    # 페이지 번호는 헤더에 "N/?" 형태 대신 페이지 번호만 표시
    # (실제 페이지 수는 analysis 길이에 따라 동적이므로 사전 계산 불가)
    total_pages = 0  # 0이면 헤더에 총 페이지 미표시

    page_num = 1

    # Page 1: 가설
    _draw_page_hypothesis(cv, W, H, FN, FB, inp, hyp, val_n, papers_list,
                          below, page_num, total_pages)

    # Page 2+: 실험 결과 (analysis가 길면 자동 페이지 넘김)
    if has_results:
        cv.showPage()
        page_num += 1
        _draw_page_results(cv, W, H, FN, FB, inp, hyp, final_summary, results,
                           page_num, total_pages)

    # 참고 문헌
    if papers_list:
        cv.showPage()
        page_num += 1
        _draw_pages_references(cv, W, H, FN, FB, papers_list, page_num, total_pages)

    cv.save()
    print(f"  PDF 저장: {output_path}  ({page_num}페이지)")


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
