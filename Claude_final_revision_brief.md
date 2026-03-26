첫째, dispatch_id는 workflow 상단 주석에서는 “입력 필수”라고 적혀 있는데, 실제 workflow_dispatch.inputs.dispatch_id는 required: false입니다. runners.py가 항상 값을 넣어주긴 하지만, 수동 실행까지 생각하면 required: true로 맞추는 편이 더 안전합니다.

둘째, 문서는 아직 최신 구조와 완전히 안 맞습니다.
현재 CLAUDE.md는 다이어그램에서 reports/, results/, experiments/를 전역 저장소처럼 보이게 그리고 있고, 모듈별 출력 표도 papers_{topic}.json, hypothesis_{topic}.json, approval_{topic}.json 같은 예전 표기를 남기고 있습니다. 실행엔 직접 영향은 없지만, 문서 신뢰성은 떨어집니다.