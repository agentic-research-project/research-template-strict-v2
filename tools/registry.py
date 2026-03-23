"""
Tool 레지스트리

Claude 오케스트레이터가 사용할 tool 스키마 정의 및 실행 라우터.
각 tool은 lab/ 모듈의 함수와 1:1 매핑된다.
"""

import json
from typing import Any


# ── Tool 스키마 정의 ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "analyze_topic",
        "description": (
            "사용자의 연구 주제를 분석하여 검색 키워드, 연구 범위, 성공 기준을 구조화한다. "
            "파이프라인의 첫 번째 단계로 반드시 먼저 호출해야 한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic":              {"type": "string", "description": "연구 주제"},
                "details":            {"type": "string", "description": "구체적 내용 (데이터 특성, 도메인 등)"},
                "problem_definition": {"type": "string", "description": "풀고자 하는 문제 정의 (현재 한계, 왜 어려운가)"},
                "desired_outcome":    {"type": "string", "description": "원하는 결과 (무엇을 달성하면 성공인가)"},
                "constraints":        {"type": "string", "description": "제약 조건 (선택)"},
                "target_metric":      {"type": "string", "description": "목표 성능 지표 (선택)"},
            },
            "required": ["topic", "details", "problem_definition", "desired_outcome"],
        },
    },
    {
        "name": "search_papers",
        "description": (
            "arXiv, Semantic Scholar에서 최신 딥러닝 논문을 검색하고 요약한다. "
            "NeurIPS, CVPR, ECCV, ICLR, ICML 등 주요 학회 논문을 우선 검색한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "primary_keywords":   {"type": "array", "items": {"type": "string"}, "description": "핵심 검색 키워드"},
                "secondary_keywords": {"type": "array", "items": {"type": "string"}, "description": "보조 검색 키워드"},
                "max_papers":         {"type": "integer", "description": "최대 논문 수 (기본 10)", "default": 10},
            },
            "required": ["primary_keywords"],
        },
    },
    {
        "name": "generate_hypothesis",
        "description": (
            "수집된 논문들을 분석하여 연구 가설을 수립한다. "
            "연구 갭을 파악하고 새로운 접근법을 제안한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic_analysis": {"type": "object", "description": "analyze_topic의 결과"},
                "papers":         {"type": "array",  "description": "search_papers의 결과"},
            },
            "required": ["topic_analysis", "papers"],
        },
    },
    {
        "name": "validate_hypothesis",
        "description": (
            "GPT-4o와 Gemini를 사용해 가설의 타당성을 검증하고 피드백을 수집한다. "
            "두 모델의 의견을 종합하여 신뢰도 점수를 산출한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "object", "description": "generate_hypothesis의 결과"},
            },
            "required": ["hypothesis"],
        },
    },
    {
        "name": "request_approval",
        "description": (
            "지금까지의 연구 결과(논문 요약, 가설, 검증 결과)를 사용자에게 보고하고 승인을 요청한다. "
            "사용자가 approve / revise / reject 중 하나를 선택한다. "
            "반드시 코드 생성 전에 호출해야 한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic_analysis":      {"type": "object", "description": "주제 분석 결과"},
                "papers":              {"type": "array",  "description": "논문 목록"},
                "hypothesis":          {"type": "object", "description": "가설"},
                "validation_result":   {"type": "object", "description": "검증 결과"},
            },
            "required": ["topic_analysis", "hypothesis", "validation_result"],
        },
    },
    {
        "name": "analyze_github_code",
        "description": (
            "관련 논문의 GitHub 구현체를 검색하고 코드 구조를 분석한다. "
            "재사용 가능한 컴포넌트와 핵심 알고리즘을 파악한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords":  {"type": "array", "items": {"type": "string"}, "description": "GitHub 검색 키워드"},
                "papers":    {"type": "array", "description": "GitHub 링크가 포함된 논문 목록"},
                "max_repos": {"type": "integer", "description": "최대 레포 수 (기본 5)", "default": 5},
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "generate_model",
        "description": (
            "가설과 코드 분석을 바탕으로 PyTorch 모델 코드를 생성한다. "
            "trainer, dataloader, loss function을 포함한 완전한 학습 코드를 작성한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis":    {"type": "object", "description": "연구 가설"},
                "code_analysis": {"type": "object", "description": "GitHub 코드 분석 결과"},
                "topic_analysis":{"type": "object", "description": "주제 분석 결과"},
            },
            "required": ["hypothesis", "code_analysis", "topic_analysis"],
        },
    },
    {
        "name": "run_experiment",
        "description": (
            "생성된 모델로 실험을 실행하고 결과를 반환한다. "
            "results/ 디렉토리에 JSON으로 저장된다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_code_path": {"type": "string", "description": "생성된 모델 코드 경로"},
                "experiment_id":   {"type": "string", "description": "실험 ID"},
            },
            "required": ["model_code_path", "experiment_id"],
        },
    },
]


# ── Tool 실행 라우터 ───────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """
    tool 이름에 따라 해당 lab 모듈 함수를 호출한다.
    """
    if tool_name == "analyze_topic":
        from lab.topic_analyzer import analyze_topic
        return analyze_topic(**tool_input)

    elif tool_name == "search_papers":
        from lab.paper_researcher import search_papers
        return search_papers(**tool_input)

    elif tool_name == "generate_hypothesis":
        from lab.hypothesis_generator import generate_hypothesis
        return generate_hypothesis(**tool_input)

    elif tool_name == "validate_hypothesis":
        from lab.hypothesis_validator import validate_hypothesis
        return validate_hypothesis(**tool_input)

    elif tool_name == "request_approval":
        from lab.user_approval import request_approval
        return request_approval(**tool_input)

    elif tool_name == "analyze_github_code":
        from lab.code_analyzer import analyze_github_code
        return analyze_github_code(**tool_input)

    elif tool_name == "generate_model":
        from lab.model_generator import generate_model
        return generate_model(**tool_input)

    elif tool_name == "run_experiment":
        from lab.research_loop import run_experiment
        return run_experiment(**tool_input)

    else:
        raise ValueError(f"Unknown tool: {tool_name}")
