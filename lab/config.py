from dotenv import load_dotenv
import anyio
import json
import os
from claude_agent_sdk import query as _sdk_query, ClaudeAgentOptions, AssistantMessage

load_dotenv()

# LLM 모델 설정
CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.1"   #5.4
GEMINI_MODEL = "gemini-2.5-pro" #3.1-pro

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")
S2_API_KEY     = os.getenv("S2_API_KEY", "")

# 검증 통과 기준 (validator + PDF 공용)
SCORE_THRESHOLD = 8.5



def get_openai_client():
    """OpenAI 클라이언트를 반환한다."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def get_gemini_model(model_name: str = GEMINI_MODEL):
    """Gemini GenerativeModel을 반환한다."""
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name)


def query_claude(prompt: str, image_paths: list[str] | None = None) -> str:
    """claude_agent_sdk를 통해 Claude에 단일 쿼리를 실행한다 (API 키 불필요).

    image_paths를 전달하면 Read 툴을 허용하여 Claude가 이미지를 직접 읽는다.
    """
    async def _run() -> str:
        full_prompt = prompt
        if image_paths:
            paths = "\n".join(f"- {p}" for p in image_paths)
            full_prompt += f"\n\n분석할 이미지 파일:\n{paths}"

        tools     = ["Read"] if image_paths else []
        max_turns = max(3, len(image_paths) + 2) if image_paths else 1

        text = ""
        async for msg in _sdk_query(
            prompt=full_prompt,
            options=ClaudeAgentOptions(
                model=CLAUDE_MODEL,
                allowed_tools=tools,
                max_turns=max_turns,
            ),
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        text += block.text
        return text
    return anyio.run(_run)


def parse_json(text: str) -> dict:
    """LLM 응답에서 마크다운 펜스를 제거하고 JSON을 파싱한다."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # ```json ... ``` 형식 처리
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text and not text.startswith("{") and not text.startswith("["):
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)
