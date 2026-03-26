"""
Stage 5: 사용자 승인 인터페이스

전 단계 결과를 PDF 보고서로 생성하고 사용자의 승인/수정/거부를 받는다.
- PDF: experiments/{slug}/reports/report.pdf  (A4 1페이지, reportlab)
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
import re
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
# reportlab 드로잉 헬퍼 (좌표계: 원점 = 페이지 좌하단)
# ──────────────────────────────────────────────────────────

def _str_width(c, text: str, font: str, size: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    return stringWidth(text, font, size)


def _wrap_lines(c, text: str, max_w: float, font: str, size: float) -> list:
    """max_w 내에서 단어 단위로 줄바꿈한 라인 목록을 반환."""
    words = str(text or "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if _str_width(c, test, font, size) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text_height(c, text: str, max_w: float, font: str, size: float,
                 line_h: float = None, max_lines: int = None) -> float:
    """텍스트 블록이 차지하는 높이(pt)를 미리 계산한다."""
    if line_h is None:
        line_h = size * 1.45
    lines = _wrap_lines(c, text, max_w, font, size)
    if max_lines:
        lines = lines[:max_lines]
    return len(lines) * line_h


def _color(color):
    """색상 문자열 → reportlab Color 객체."""
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
    """
    텍스트를 max_w 내에서 자동 줄바꿈하여 그린다.
    마지막 라인의 baseline y 좌표를 반환한다.
    """
    if line_h is None:
        line_h = size * 1.45
    lines = _wrap_lines(c, text, max_w, font, size)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while _str_width(c, last + "...", font, size) > max_w and last:
            last = last[:-1]
        lines[-1] = last + "..."
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
    """채워진 (선택적 테두리) 사각형."""
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
    """수평 화살표 (x1 → x2)."""
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
    """작은 라벨 칩 (배경 + 텍스트)."""
    _rect(c, x, y, w, h, bg, radius=2)
    c.setFont(font, size)
    c.setFillColor(_color(color))
    c.drawString(x + 2, y + (h - size) / 2 + 1, text)


def _find_paper(papers_list: list, hint) -> dict:
    """papers_list에서 hint와 제목이 유사한 논문을 찾아 반환."""
    h = (hint.get("title", "") if isinstance(hint, dict) else hint).lower()
    for p in papers_list:
        if any(w in p.get("title", "").lower() for w in h.split()[:3]):
            return p
    return None


def _draw_bar(c, x: float, by: float, bar_w: float, bar_h: float,
              score: float, verdict: str, label: str,
              bar_color_map: dict, font_reg: str, font_bold: str, lbl_size: float):
    """LLM 점수 한 행 (레이블 + 배경바 + 채우기바 + 점수) 그리기."""
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
    """Pipeline 단계 박스 (배경 + 2줄 레이블) 그리기."""
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
    """불릿 항목 목록을 그리고 마지막 y 좌표를 반환."""
    ry = y
    for item in items:
        ry = _text(c, f"• {item}", x, ry, w, font, size, color, line_h=item_lh)
        ry -= item_gap
    return ry


# ──────────────────────────────────────────────────────────
# PDF 생성
# ──────────────────────────────────────────────────────────

def generate_pdf(
    topic: dict,
    hypothesis: dict,
    papers: dict,
    validation: dict,
    output_path: Path,
) -> None:
    """reportlab A4 1페이지 PDF 생성 — 텍스트 길이 무관하게 자동 배치."""
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white

    # ── 데이터 추출 ──────────────────────────────────────────
    inp         = topic.get("input", {})
    hyp         = hypothesis.get("hypothesis", {})
    gap         = hypothesis.get("research_gap", {})
    exp_plan    = hypothesis.get("experiment_plan", {})
    risks       = hypothesis.get("risk_factors", [])
    papers_list = papers.get("papers", [])
    related     = hypothesis.get("related_papers", [])
    summary     = validation.get("summary", {})
    gpt_val     = validation.get("gpt4o", {})
    gem_val     = validation.get("gemini", {})
    avg         = summary.get("average_score", 0)
    below       = avg < PDF_SCORE_THRESHOLD

    # 근거 논문 3편 선택 (_find_paper는 모듈 수준 함수)
    top_papers = []
    for rt in related:
        p = _find_paper(papers_list, rt)
        if p and p not in top_papers:
            top_papers.append(p)
    for p in papers_list:
        if len(top_papers) >= 3:
            break
        if p not in top_papers:
            top_papers.append(p)
    top_papers = (top_papers + [None, None, None])[:3]

    # ── 폰트 & 캔버스 ─────────────────────────────────────
    FN, FB = _setup_fonts()
    W, H   = A4   # 595.27 x 841.89 pt
    output_path.parent.mkdir(exist_ok=True)
    c = rl_canvas.Canvas(str(output_path), pagesize=A4)

    ML = 8 * mm          # 좌측 마진
    CW = W - 2 * ML      # 컨텐츠 폭

    # ── 페이지 예산 (A4 1페이지 고정) ───────────────────────
    # 6개 섹션 × 고정 비율, 초과 텍스트는 max_lines 로 자름
    _GAP   = 3 * mm
    _NET_H = H - 14 * mm - 3 * mm - ML - 5 * _GAP  # 헤더·하단마진·갭 제외 순가용 높이
    _B = {
        "hyp":     _NET_H * 0.17,   # 가설 + LLM 점수
        "pipe":    _NET_H * 0.14,   # 파이프라인
        "novelty": _NET_H * 0.13,   # 핵심혁신 + 작동원리
        "papers":  _NET_H * 0.16,   # 근거 논문
        "exp":     _NET_H * 0.32,   # 실험계획 + 리스크
        "verdict": _NET_H * 0.08,   # 판정 박스
    }
    FTR_H = _B["verdict"]           # 판정 박스 고정 높이 (exp_avail 계산에 미리 필요)

    # ══════════════════════════════════════════════════════
    # 0. HEADER (상단 고정)
    # ══════════════════════════════════════════════════════
    HDR_H = 14 * mm
    _rect(c, 0, H - HDR_H, W, HDR_H, C_BG)
    c.setFillColor(white)
    c.setFont(FB, 16)
    c.drawCentredString(W / 2, H - HDR_H + HDR_H * 0.60, "Research Hypothesis Report")
    c.setFont(FN, 8)
    c.setFillColor(HexColor(C_GRAY))
    c.drawCentredString(
        W / 2, H - HDR_H + HDR_H * 0.22,
        f"주제: {inp.get('topic', 'Research')}   |   {datetime.now().strftime('%Y-%m-%d')}",
    )

    CHIP_OFFSET = 0.5 * mm  # 칩 하단 y 미세 조정 (칩이 섹션 상단 경계에 걸치도록)

    # 현재 Y (아래로 내려가며 배치)
    cur_y = H - HDR_H - 3 * mm

    # ══════════════════════════════════════════════════════
    # 1. HYPOTHESIS (왼쪽 65%) + LLM SCORE (오른쪽 32%)
    # ══════════════════════════════════════════════════════
    CHIP_H   = 5 * mm    # 섹션 헤더 칩 높이
    CHIP_GAP = 4 * mm    # 칩 아래 → 텍스트 상단 간격
    PAD      = 3 * mm    # 섹션 상하 여백

    HYP_W = CW * 0.65
    LLM_X = ML + HYP_W + CW * 0.03
    LLM_W = CW - HYP_W - CW * 0.03

    stmt_kr  = hyp.get("statement_kr", hyp.get("statement", ""))
    innov    = hyp.get("key_innovation", "")
    innov_lh = 8.5   # 핵심 혁신 줄 높이 (pt)

    # LLM 바 3개 높이
    bar_h      = 6 * mm
    bar_gap    = 6.5 * mm
    llm_bars_h = 3 * (bar_h + bar_gap)

    # HYP_H = 예산 고정 (LLM 최소 높이 보장)
    HYP_H = max(_B["hyp"], CHIP_H + CHIP_GAP + llm_bars_h + PAD)

    # 핵심 혁신 박스: HYP_H의 28% 이내로 제한
    innov_content_h = _text_height(c, "핵심 혁신: " + innov, HYP_W - 8*mm, FN, 6.5, line_h=innov_lh)
    INNOV_H = min(innov_content_h + 2 * PAD, HYP_H * 0.28)
    innov_max_ln = max(1, int((INNOV_H - 2 * PAD) / innov_lh))

    # 가설 텍스트: 남은 공간에서 max_lines 유도
    stmt_avail  = HYP_H - CHIP_H - CHIP_GAP - PAD - INNOV_H - PAD
    stmt_max_ln = max(2, int(stmt_avail / 11.5))

    hyp_top = cur_y
    hyp_bot = cur_y - HYP_H

    # ── 가설 배경 & 헤더 칩 ──────────────────────────────
    _rect(c, ML, hyp_bot, HYP_W, HYP_H, C_HYPO, radius=3)
    chip_bottom = hyp_top - CHIP_H - CHIP_OFFSET   # 칩 하단 y
    _chip(c, ML + 2 * mm, chip_bottom, 50 * mm, CHIP_H,
          C_LIGHT, "핵심 가설 (Core Hypothesis)", FB, 6.5, C_HYPO)

    # ── 가설 본문: 칩 아래 CHIP_GAP 여백 후 시작 ─────────
    text_top = chip_bottom - CHIP_GAP
    text_end = _text(c, stmt_kr,
                     ML + 4 * mm, text_top, HYP_W - 8 * mm,
                     FB, 7.5, "white", line_h=11.5, max_lines=stmt_max_ln)

    # ── 핵심 혁신: HYP_H가 정확히 맞춰져 있으므로 박스 하단 기준으로 배치
    innov_top = hyp_bot + PAD + INNOV_H
    innov_bot = innov_top - INNOV_H
    _rect(c, ML + 2 * mm, innov_bot, HYP_W - 4 * mm, INNOV_H, "#1A1A2E", radius=2)
    _text(c, "핵심 혁신: " + innov,
          ML + 4 * mm, innov_bot + INNOV_H - PAD, HYP_W - 8 * mm,
          FN, 6.5, C_GOLD, line_h=innov_lh, max_lines=innov_max_ln)

    # ── LLM 검증 바 차트: 칩 하단 + CHIP_GAP 후 시작 ──────
    llm_chip_bottom = hyp_top - CHIP_H - CHIP_OFFSET
    _chip(c, LLM_X, llm_chip_bottom, 22 * mm, CHIP_H,
          C_LIGHT, "LLM 검증", FB, 6.5, C_HYPO)

    bar_color_map = {"approve": C_APPROVE, "revise": C_WARN, "reject": C_ACCENT}
    bar_max_w  = LLM_W - 16 * mm
    lbl_size   = 6.5   # pt (바 레이블 폰트 크기)
    # 첫 번째 바: 칩 하단 → CHIP_GAP → 레이블 텍스트 높이 → 바 시작
    bar_start  = llm_chip_bottom - CHIP_GAP - lbl_size - bar_h

    bar_data = [
        ("GPT-4o", gpt_val.get("score", 0), gpt_val.get("verdict", "")),
        ("Gemini", gem_val.get("score", 0), gem_val.get("verdict", "")),
        ("종합",   avg,                      summary.get("overall_verdict", "")),
    ]
    for i, (lbl, score, verdict) in enumerate(bar_data):
        by = bar_start - i * (bar_h + bar_gap)
        _draw_bar(c, LLM_X, by, bar_max_w, bar_h,
                  score, verdict, lbl, bar_color_map, FN, FB, lbl_size)

    c.setFont(FN, 6)
    c.setFillColor(HexColor(C_GRAY))
    c.drawCentredString(LLM_X + bar_max_w / 2, hyp_bot + 3 * mm, "점수 / 10")

    cur_y = hyp_bot - 3 * mm

    # ══════════════════════════════════════════════════════
    # 2. RESEARCH PIPELINE
    # ══════════════════════════════════════════════════════
    # Pipeline 섹션 높이: 헤더 공간 + 박스 + 박스-서브텍스트 갭 + 서브텍스트 + 하단 여백
    box_h        = 16 * mm
    pipe_hdr     = 8 * mm    # 칩 헤더 공간 (pipe_top 기준)
    sub_gap      = 2 * mm    # box_bot → 서브텍스트 사이 간격
    sub_lh       = 7         # 서브텍스트 줄 높이 (pt, reportlab 좌표계와 동일 단위)
    pipe_bot_pad = 3 * mm

    # PIPE_H = 예산 고정, sub-text는 남은 공간에서 max_lines 유도
    lims    = gap.get("limitations", [])
    gap_sum = gap.get("summary", "")
    metric  = ", ".join(inp.get("target_metric", "PSNR, SSIM").split(",")[:2]).strip()
    _pipe_box_w = CW * 0.185
    PIPE_H      = _B["pipe"]
    sub_avail   = PIPE_H - pipe_hdr - box_h - sub_gap - pipe_bot_pad
    sub_max_ln  = max(1, int(sub_avail / sub_lh))
    sub_block_h = sub_max_ln * sub_lh

    pipe_top = cur_y
    pipe_bot = cur_y - PIPE_H

    _rect(c, ML, pipe_bot, CW, PIPE_H, "#F8F9FA", radius=3)
    _chip(c, ML + 2 * mm, pipe_top - 5.5 * mm, 38 * mm, 5 * mm,
          C_LIGHT, "Research Pipeline", FB, 6.5, C_HYPO)

    # 서브텍스트: sub_block_h 내에서 최대 줄 수 동적 계산 (고정 max_lines 사용 금지)
    sub_max_ln = max(1, int(sub_block_h / sub_lh))
    stages    = [
        ("기존 방법\n(Existing)",  C_GRAY,    lims[0] if lims else ""),
        ("연구 갭\n(Gap)",          C_ACCENT,  gap_sum),
        ("제안 방법\n(Proposed)",   C_HYPO,    hyp.get("statement_kr", "")),
        ("예상 성과\n(Outcome)",    C_APPROVE, metric),
    ]
    box_w      = _pipe_box_w
    box_spc    = (CW - 4 * box_w) / 3
    lbl_line_h = 4.5 * mm   # 박스 내 레이블 줄 간격
    box_top    = pipe_top - pipe_hdr   # 박스 상단
    box_bot    = box_top - box_h       # 박스 하단
    sub_top    = box_bot - sub_gap     # 서브텍스트 시작 (박스 하단 기준)

    for i, (label, color, sub) in enumerate(stages):
        bx = ML + i * (box_w + box_spc)
        _draw_stage_box(c, bx, box_bot, box_w, box_h, label, color, lbl_line_h, FB, 7)
        # 화살표
        if i < 3:
            ax1 = bx + box_w + 1 * mm
            ax2 = bx + box_w + box_spc - 1 * mm
            _arrow(c, ax1, box_bot + box_h / 2, ax2, color=C_GRAY)
        # 서브텍스트: box_bot 기준 아래 배치, max_lines로 줄 수 제한
        _text(c, sub, bx + 1 * mm, sub_top, box_w - 2 * mm,
              FN, 5.5, color, line_h=sub_lh, max_lines=sub_max_ln, align="center")

    cur_y = pipe_bot - 3 * mm

    # ══════════════════════════════════════════════════════
    # 3. NOVELTY  (key_innovation 왼쪽 / expected_mechanism 오른쪽)
    # ══════════════════════════════════════════════════════
    key_innov = hyp.get("key_innovation", "")
    mechanism = hyp.get("expected_mechanism", "")

    nov_gap  = CW * 0.03                        # 두 패널 사이 간격 (컨텐츠 폭의 3%)
    nov_w    = (CW - nov_gap) / 2               # 좌/우 패널 폭
    nov_lh   = 8.5                              # 본문 줄 높이 (pt)
    # NOV_H = 예산 고정, 텍스트는 남은 공간에서 max_lines 유도
    NOV_H      = _B["novelty"]
    nov_max_ln = max(2, int((NOV_H - CHIP_H - CHIP_GAP - PAD) / nov_lh))

    nov_top  = cur_y
    nov_bot  = cur_y - NOV_H
    nov_rx   = ML + nov_w + nov_gap             # 오른쪽 패널 x

    _rect(c, ML,     nov_bot, nov_w, NOV_H, "#EEF4FF", radius=3)
    _rect(c, nov_rx, nov_bot, nov_w, NOV_H, "#F0FFF4", radius=3)

    _chip(c, ML     + 2*mm, nov_top - CHIP_H - CHIP_OFFSET, 44*mm, CHIP_H,
          C_HYPO,    "핵심 혁신 (Key Innovation)", FB, 6.5, "white")
    _chip(c, nov_rx + 2*mm, nov_top - CHIP_H - CHIP_OFFSET, 38*mm, CHIP_H,
          C_APPROVE, "작동 원리 (Mechanism)",      FB, 6.5, "white")

    text_start = nov_top - CHIP_H - CHIP_GAP
    _text(c, key_innov, ML     + 3*mm, text_start, nov_w - 6*mm,
          FN, 6.5, "#1A3A6B", line_h=nov_lh, max_lines=nov_max_ln)
    _text(c, mechanism, nov_rx + 3*mm, text_start, nov_w - 6*mm,
          FN, 6.5, "#1A5C3A", line_h=nov_lh, max_lines=nov_max_ln)

    cur_y = nov_bot - 3 * mm

    # ══════════════════════════════════════════════════════
    # 4. 근거 논문 (Supporting Evidence)
    # ══════════════════════════════════════════════════════
    c.setFont(FB, 8)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(ML, cur_y - 4.5 * mm, "근거 논문 (Supporting Evidence)")
    cur_y -= 8 * mm

    card_w   = (CW - 2 * 3 * mm) / 3
    card_gap = 3 * mm

    # 카드 높이: 각 논문의 제목+저자 실제 높이를 미리 계산 → 최대값 + abstract 공간 + 뱃지 공간
    def _card_header_h(paper):
        if paper is None:
            return 0.0
        t_h = _text_height(c, paper.get("title", ""), card_w - 6*mm, FB, 7, line_h=10, max_lines=3)
        authors = ", ".join(paper.get("authors", [])[:2])
        year    = paper.get("year", "")
        venue   = paper.get("venue", paper.get("source", ""))
        meta    = f"{authors}  ({year})" + (f"  [{venue}]" if venue else "")
        a_h = _text_height(c, meta, card_w - 6*mm, FN, 6, line_h=8.5, max_lines=2)
        return 5*mm + t_h + 3*mm + a_h + 2*mm   # 상단여백 + 제목 + 갭 + 저자 + 갭

    badge_r      = 2.5 * mm                    # 번호 뱃지 반지름
    badge_space  = badge_r * 2 + 3*mm + 2*mm   # 뱃지 지름 + 하단여백 + 추가여백
    abstract_lh  = 8                            # abstract 줄 높이 (pt, 좌표계 동일 단위)
    abstract_min = 4 * abstract_lh              # 최소 4줄 확보 (pt)
    max_header_h = max((_card_header_h(p) for p in top_papers), default=0.0)
    # PAP_H: 최소 (badge+abstract+여백), 실제 헤더가 크면 그에 맞게 확장
    PAP_H = min(
        max(badge_space + abstract_min + PAD * 2,
            max_header_h + abstract_min + badge_space),
        _B["papers"],
    )

    pap_bot = cur_y - PAP_H

    for col_i, paper in enumerate(top_papers):
        cx = ML + col_i * (card_w + card_gap)
        cy = pap_bot
        if paper is None:
            _rect(c, cx, cy, card_w, PAP_H, "#EEF0F2", radius=3)
            c.setFont(FN, 8)
            c.setFillColor(HexColor(C_GRAY))
            c.drawCentredString(cx + card_w / 2, cy + PAP_H / 2, "논문 없음")
            continue
        _rect(c, cx, cy, card_w, PAP_H, C_CARD, radius=3)
        # 번호 뱃지 — 오른쪽 하단에 배치 (badge_r는 섹션 상단에서 정의됨)
        badge_x = cx + card_w - 3 * mm - 2 * badge_r
        badge_y = cy + 3 * mm
        _rect(c, badge_x, badge_y, 2 * badge_r, 2 * badge_r, C_ACCENT, radius=badge_r)
        c.setFont(FB, 5.5)
        c.setFillColor(white)
        c.drawCentredString(badge_x + badge_r, badge_y + badge_r - 0.8, str(col_i + 1))
        # 제목 → 반환값을 다음 요소 시작점으로 사용 (flowing layout)
        title_end = _text(c, paper.get("title", ""),
                          cx + 3 * mm, cy + PAP_H - 5 * mm, card_w - 6 * mm,
                          FB, 7, "white", line_h=10, max_lines=3)
        # 저자 + 연도 + 학회: 제목 끝 기준 3mm 아래
        authors = ", ".join(paper.get("authors", [])[:2])
        year    = paper.get("year", "")
        venue   = paper.get("venue", paper.get("source", ""))
        meta    = f"{authors}  ({year})"
        if venue:
            meta += f"  [{venue}]"
        author_end = _text(c, meta,
                           cx + 3 * mm, title_end - 3 * mm, card_w - 6 * mm,
                           FN, 6, C_GOLD, line_h=8.5, max_lines=2)
        # 요약: 저자 끝 기준 2mm 아래, 뱃지 상단 기준 최대 줄 수 동적 계산
        abstract_top    = author_end - 2 * mm
        badge_top       = badge_y + 2 * badge_r        # 뱃지 상단 절대 y
        abstract_max_h  = abstract_top - badge_top - 2 * mm
        abstract_max_ln = max(1, int(abstract_max_h / abstract_lh))  # pt/pt → 줄 수
        _text(c, paper.get("abstract", ""),
              cx + 3 * mm, abstract_top, card_w - 6 * mm,
              FN, 5.8, C_GRAY, line_h=abstract_lh, max_lines=abstract_max_ln)

    cur_y = pap_bot - 3 * mm

    # ══════════════════════════════════════════════════════
    # 4. 실험 계획 (왼쪽 63%) + 근거 & 리스크 (오른쪽 34%)
    # ══════════════════════════════════════════════════════
    EXP_W  = CW * 0.63
    RISK_X = ML + EXP_W + CW * 0.03
    RISK_W = CW - EXP_W - CW * 0.03

    baselines = exp_plan.get("baseline_models", [])
    metrics   = exp_plan.get("evaluation_metrics", [])
    exp_rows  = [
        ("아키텍처", exp_plan.get("architecture", "")),
        ("데이터셋",  exp_plan.get("dataset", "")),
        ("기준 모델", "  /  ".join(b.split("(")[0].strip() for b in baselines[:4])),
        ("평가 지표", "  /  ".join(m.split("—")[0].strip() for m in metrics[:4])),
    ]
    lbl_col_w = 18 * mm              # 레이블([아키텍처] 등) 열 폭
    lbl_x     = ML + PAD            # 레이블 x 시작
    val_x     = lbl_x + lbl_col_w  # 값 텍스트 x 시작
    val_w     = EXP_W - lbl_col_w - 2 * PAD  # 값 텍스트 최대 폭
    row_gap   = 2 * mm
    exp_hdr_h = 9 * mm   # 칩 헤더 공간

    strengths   = summary.get("all_strengths", [])
    item_gap    = 1.5 * mm
    lbl_gap     = 5 * mm
    section_gap = 5 * mm
    risk_hdr_h  = 10 * mm

    # ── 판정 박스 텍스트 준비 (FTR_H = _B["verdict"] 고정, 초과 텍스트는 max_lines 로 자름)
    vrd_main_size   = 8   # pt
    vrd_sub_size    = 6   # pt
    overall         = summary.get("overall_verdict", "")
    if below:
        vrd_main_text = (
            f"[!] 점수 미달  {avg}/10  (목표 {PDF_SCORE_THRESHOLD}점) — 사용자 판단 필요"
        )
        vrd_sub_text = f"주요 약점: {summary.get('all_weaknesses', ['약점 정보 없음'])[0]}"
    else:
        v_label = {"approve": "[OK] 승인 권고",
                   "revise":  "[!] 수정 권고",
                   "reject":  "[X] 거부 권고"}.get(overall, overall)
        vrd_main_text = f"LLM 종합 판정:  {v_label}   (평균 {avg} / 10)"
        vrd_sub_text  = "AI-Driven Research Automation  |  Generated by Claude Opus"
    vrd_text_w      = EXP_W - 4 * PAD
    vrd_half_h      = (FTR_H - 3 * PAD) / 2          # 두 줄에 균등 배분
    vrd_main_max_ln = max(1, int(vrd_half_h / (vrd_main_size * 1.45)))
    vrd_sub_max_ln  = max(1, int(vrd_half_h / (vrd_sub_size  * 1.45)))

    # 실험계획(좌)은 exp_avail 내에서 폰트 자동 축소
    # 근거&리스크(우)는 RISK_H(페이지 하단까지)가 독립 높이이므로 루프 조건에서 분리
    exp_avail      = cur_y - FTR_H - PAD - ML
    exp_base_size  = 9.0   # 아키텍처/데이터셋 등 내용 폰트 기본 크기
    item_base_size = 7.5   # 근거/리스크 항목 폰트 기본 크기
    scale = 1.0
    exp_size = exp_base_size
    item_size = item_base_size
    exp_content_h = risk_content_h = 0.0   # 초기화

    while scale >= 0.50:
        exp_size  = round(exp_base_size  * scale, 2)
        item_size = round(item_base_size * scale, 2)
        row_lh    = exp_size  * 1.45
        item_lh   = item_size * 1.45

        exp_content_h = exp_hdr_h + sum(
            _text_height(c, val, val_w, FN, exp_size, line_h=row_lh) + row_gap
            for _, val in exp_rows
        )
        strengths_h = sum(
            _text_height(c, f"• {s}", RISK_W - 6*mm, FN, item_size, line_h=item_lh) + item_gap
            for s in strengths[:2]
        )
        risks_h = sum(
            _text_height(c, f"• {r}", RISK_W - 6*mm, FN, item_size, line_h=item_lh) + item_gap
            for r in risks[:3]
        )
        risk_content_h = risk_hdr_h + lbl_gap + strengths_h + section_gap + lbl_gap + risks_h

        # 실험계획(좌)만 exp_avail 기준으로 축소 — 근거&리스크(우)는 RISK_H로 독립 확장
        if exp_content_h <= exp_avail:
            break
        scale = round(scale - 0.05, 2)

    # EXP_H는 실험계획 기준, exp_avail 초과 방지 (판정 박스가 페이지 밖으로 밀리지 않도록)
    EXP_H = min(exp_content_h, exp_avail)
    exp_top = cur_y
    exp_bot = cur_y - EXP_H

    # 실험 계획 배경
    _rect(c, ML, exp_bot, EXP_W, EXP_H, C_LIGHT, radius=3)
    _chip(c, ML + 2 * mm, exp_top - 5.5 * mm, 54 * mm, 5 * mm,
          C_HYPO, "실험 계획 (Experiment Plan)", FB, 6.5, "white")

    row_y = exp_top - exp_hdr_h
    for lbl, val in exp_rows:
        c.setFont(FB, 7)
        c.setFillColor(HexColor(C_HYPO))
        c.drawString(lbl_x, row_y, f"[{lbl}]")
        next_y = _text(c, val, val_x, row_y, val_w,
                       FN, exp_size, "#333333", line_h=row_lh)
        row_y = next_y - row_gap

    # 근거 & 리스크 배경 — 페이지 하단(ML)까지 확장
    RISK_H = cur_y - ML   # cur_y == exp_top, ML = 하단 마진
    _rect(c, RISK_X, ML, RISK_W, RISK_H, "#FFF8EE", radius=3)
    _chip(c, RISK_X + 2 * mm, exp_top - 5.5 * mm, 26 * mm, 5 * mm,
          C_HYPO, "근거 & 리스크", FB, 6.5, "white")

    # 근거 (validation 강점)
    c.setFont(FB, 7)
    c.setFillColor(HexColor(C_HYPO))
    c.drawString(RISK_X + 3 * mm, exp_top - risk_hdr_h, "근거")
    ry = _draw_bullet_list(c, strengths[:2],
                           RISK_X + 3 * mm, exp_top - risk_hdr_h - lbl_gap,
                           RISK_W - 6 * mm, FN, item_size, "#222222", item_lh, item_gap)

    # 리스크: 근거 끝 기준 flowing 배치
    c.setFont(FB, 7)
    c.setFillColor(HexColor(C_ACCENT))
    c.drawString(RISK_X + 3 * mm, ry - section_gap, "리스크")
    _draw_bullet_list(c, risks[:3],
                      RISK_X + 3 * mm, ry - section_gap - lbl_gap,
                      RISK_W - 6 * mm, FN, item_size, "#555555", item_lh, item_gap)

    cur_y = exp_bot - PAD

    # ══════════════════════════════════════════════════════
    # 5. VERDICT (실험계획 바로 아래 인라인 배치)
    # ══════════════════════════════════════════════════════
    vrd_bot = cur_y - FTR_H
    vrd_top = vrd_bot + FTR_H  # 박스 상단

    if below:
        _rect(c, ML, vrd_bot, EXP_W, FTR_H, "#7D3C00", radius=3)
        y = _text(c, vrd_main_text,
                  ML + 2 * PAD, vrd_top - PAD, vrd_text_w,
                  FB, vrd_main_size, "#FFD580", align="center",
                  max_lines=vrd_main_max_ln)
        _text(c, vrd_sub_text,
              ML + 2 * PAD, y - PAD, vrd_text_w,
              FN, vrd_sub_size, "#FFECB3", align="center",
              max_lines=vrd_sub_max_ln)
    else:
        v_color = {"approve": C_APPROVE, "revise": C_WARN,
                   "reject": C_ACCENT}.get(overall, "white")
        _rect(c, ML, vrd_bot, EXP_W, FTR_H, C_BG, radius=3)
        y = _text(c, vrd_main_text,
                  ML + 2 * PAD, vrd_top - PAD, vrd_text_w,
                  FB, vrd_main_size, v_color, align="center",
                  max_lines=vrd_main_max_ln)
        _text(c, vrd_sub_text,
              ML + 2 * PAD, y - PAD, vrd_text_w,
              FN, vrd_sub_size, C_GRAY, align="center",
              max_lines=vrd_sub_max_ln)

    c.save()
    print(f"  PDF 저장: {output_path}")


# ──────────────────────────────────────────────────────────
# 승인 요청
# ──────────────────────────────────────────────────────────

def request_approval(
    topic_file: str,
    hypothesis_file: str,
    validation_file: str,
    papers_file: str = "",
) -> dict:
    topic      = _load_json(topic_file)
    hypothesis = _load_json(hypothesis_file)
    validation = _load_json(validation_file)
    papers     = _load_json(papers_file) if papers_file else {}

    # validation에 최고 점수 가설이 있으면 hypothesis + 전체 연구 패키지 교체
    # validation["hypothesis"]는 전체 hypothesis 파일 구조이므로 한 겹 풀어서 교체
    _PACKAGE_FIELDS = ("research_gap", "experiment_plan", "related_papers",
                       "risk_factors", "evidence_pack", "falsification_criteria",
                       "evidence_links")
    if "hypothesis" in validation:
        val_hyp = validation["hypothesis"]
        if isinstance(val_hyp, dict) and "hypothesis" in val_hyp:
            hypothesis["hypothesis"] = val_hyp["hypothesis"]
            # 최신 개선 가설에 맞춰 전체 연구 패키지 동기화
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

    # 항상 PDF 생성 (점수 무관)
    pdf_path = reports_dir / "report.pdf"
    print("\n  PDF 보고서 생성 중...")
    generate_pdf(topic, hypothesis, papers, validation, pdf_path)

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
    args = parser.parse_args()

    result = request_approval(
        args.topic_file,
        args.hypothesis_file,
        args.validation_file,
        args.papers_file,
    )
    print(f"\n최종 결정: {result['decision']}")
