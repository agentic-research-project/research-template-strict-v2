---
name: git_init
description: Use this skill when the user types /git_init or wants to initialize a new git repository and create a GitHub repo from scratch. Asks the user for repo name and visibility, then runs git init, creates the GitHub repo via API, sets up remote, and makes the first commit + push. Always invoke for any "새 repo 만들어줘", "git 초기화", "github에 새 프로젝트 올려줘" type requests.
---

# git_init 스킬

새 프로젝트를 GitHub에 올리기 위한 전체 초기화 작업을 수행한다.

## 실행 순서

### 1단계 — 정보 수집 (AskUserQuestion)

아래 3가지를 **한 번에** 질문한다:

```
GitHub repo 설정을 알려주세요:

1. repo 이름: (현재 폴더명: {현재폴더명})
   그냥 Enter 치면 현재 폴더명으로 설정됩니다.

2. 공개 여부: private / public (기본값: private)

3. repo 설명 (선택): 한 줄 설명 또는 Enter로 건너뜀
```

### 2단계 — .gitignore 확인

- 프로젝트 루트에 `.gitignore`가 없으면 Python 기본 `.gitignore`를 자동 생성한다:

```
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
venv/
.DS_Store
*.egg-info/
dist/
build/
```

- 이미 있으면 그대로 사용한다.

### 3단계 — GitHub repo 생성

`.env` 파일에서 `GITHUB_TOKEN`을 읽어 GitHub API로 repo를 생성한다:

```bash
TOKEN=$(grep GITHUB_TOKEN .env | cut -d= -f2)

curl -s -X POST https://api.github.com/user/repos \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"{repo_name}\",\"private\":{true/false},\"description\":\"{description}\"}"
```

응답에서 `html_url`과 `clone_url`을 추출한다.
- 실패하면 에러 메시지를 출력하고 중단한다.

### 4단계 — git 초기화 및 remote 연결

```bash
git init
git branch -m main
git config user.email "$(git config --global user.email 2>/dev/null || echo 'dev@local')"
git config user.name "$(git config --global user.name 2>/dev/null || echo 'developer')"
```

remote URL에 토큰을 포함하여 인증 문제를 방지한다:
```bash
TOKEN=$(grep GITHUB_TOKEN .env | cut -d= -f2)
GITHUB_USER=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])")
git remote add origin "https://${TOKEN}@github.com/${GITHUB_USER}/{repo_name}.git"
```

### 5단계 — 첫 커밋 & push

`.env` 파일이 스테이징되지 않도록 반드시 확인한 후 진행한다:

```bash
git add -A
git status --short   # .env 포함 여부 확인
git commit -m "Initial commit: {repo_name}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push -u origin main
```

### 6단계 — 완료 보고

성공 시 다음을 출력한다:
```
✅ GitHub repo 생성 완료
   이름:   {repo_name}
   URL:    https://github.com/{user}/{repo_name}
   공개:   private / public
   파일:   N개 업로드
```

## 주의사항

- `.env`가 `git add -A` 에 포함될 경우 **즉시 중단**하고 `.gitignore`에 `.env` 추가를 안내한다
- `GITHUB_TOKEN`이 `.env`에 없으면 환경변수에서 찾고, 둘 다 없으면 토큰 입력을 요청한다
- 이미 `git init`이 되어 있는 폴더면 `git init` 단계를 건너뛴다
- 이미 `origin` remote가 있으면 사용자에게 알리고 덮어쓸지 확인한다
