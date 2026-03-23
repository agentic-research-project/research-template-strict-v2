---
name: git_push
description: Use this skill when the user types /git_push or asks to commit and push code to GitHub. Automatically stages all changes, generates a concise Korean/English commit message from the diff, commits, and pushes to origin main. Always invoke this skill for any git commit + push workflow, even if the user just says "올려줘", "push해줘", "커밋해줘", or "업로드해".
---

# git_push 스킬

변경된 코드를 GitHub에 자동으로 커밋하고 푸시한다.

## 실행 순서

### 1단계 — 변경 사항 파악
두 명령을 **동시에** 실행한다:
- `git status` — 변경/추가/삭제 파일 목록
- `git diff HEAD` — 실제 변경 내용

### 2단계 — 커밋 메시지 생성
diff 내용을 보고 **아래 규칙**으로 커밋 메시지를 작성한다:

- 제목(1줄): `type: 핵심 변경 요약` (영어, 50자 이내)
  - type 예시: `fix`, `feat`, `refactor`, `docs`, `chore`
- 본문(선택): 변경 이유나 영향이 명확히 설명이 필요한 경우만 추가
- 불필요한 설명, 파일 목록 나열 금지

좋은 예:
```
fix: remove hardcoded PSNR/SSIM metrics in model_generator
feat: add dynamic file filter keywords in code_analyzer
refactor: replace fixed FTR_H with budget-based layout
```

### 3단계 — 스테이징 & 커밋 & 푸시
아래 순서로 실행한다:

```bash
git add -A
git commit -m "<생성한 커밋 메시지>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
```

### 4단계 — 결과 보고
푸시 성공 시 다음을 출력한다:
- 커밋 해시 (short)
- 커밋 메시지
- 변경된 파일 수 / 추가 라인 / 삭제 라인

## 주의사항

- `.env` 파일이 스테이징되면 **즉시 중단**하고 사용자에게 경고한다
- `git push` 실패 시 에러 메시지를 그대로 출력하고 원인을 설명한다
- 변경 사항이 없으면 "커밋할 변경 사항이 없습니다"라고 알린다
