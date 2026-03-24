# System Design: Spec-Driven Multi-Model Experiment Loop

## 1. SYSTEM_OVERVIEW

```mermaid
flowchart TD
    A([approval.json]) --> B[code_analyzer]
    B -->|code_analysis.json| C[experiment_spec_generator\nClaude]
    C -->|experiment_spec.json| D[multi-model proposal round]

    subgraph Proposal["Multi-Model Proposal / Review / Merge"]
        D1[GPT/Codex\ncode patch proposals] --> M
        D2[Gemini\ndesign review] --> M
        M[Claude\nfinal merge & integrate]
    end

    D --> Proposal
    M -->|experiments/{slug}_v{N}/| E[GitLab CI/CD]

    subgraph GitLab["GitLab Execution"]
        E1[setup] --> E2[smoke_test]
        E2 --> E3[train]
        E3 -->|stdout: METRICS:{...}| E4[package artifacts]
    end

    E --> GitLab
    E4 -->|result_summary.json| F[Revision Decision\nClaude]

    F -->|Path A| G[Revise experiment/code]
    F -->|Path B| H[Refine hypothesis]
    F -->|Path C| I[Replace hypothesis]

    G --> C
    H -->|updated hypothesis.json| C
    I -->|new hypothesis.json| B

    F -->|done| Done([Final Report])
```

**Ownership summary:**
| Stage | Owner | Writes |
|---|---|---|
| Spec generation | Claude | experiment_spec.json |
| Code proposal | GPT/Codex | proposals/gpt_*.diff |
| Design review | Gemini | proposals/gemini_*.json |
| Final merge | Claude | experiments/{pkg}/ |
| Execution | GitLab CI | result_summary.json, artifacts/ |
| Revision decision | Claude | revision_request.json |

---

## 2. CANONICAL_INPUTS

| File | Mutable? | Owner | Source-of-truth at |
|---|---|---|---|
| `hypothesis.json` | Path B/C only | Claude (via validator) | Stages 3-8 |
| `experiment_plan.json` | Path A/B | Claude | Code generation |
| `code_analysis.json` | Read-only reference | code_analyzer | Code generation only |
| `approval.json` | Immutable after gate | user_approval | All post-approval stages |
| `previous_results.jsonl` | Append-only | GitLab CI | Revision decision |
| `experiments/claude.md` | Versioned | Claude | All code generation |

**Update rules:**
- `hypothesis.json`: frozen in Path A; refined in Path B (preserve `hypothesis_id`, add `refined_from`); replaced in Path C (new `hypothesis_id`, link old via `supersedes`)
- `experiment_plan.json`: always regenerated when hypothesis changes; also updated in Path A when experiment design is the problem
- `code_analysis.json`: never overwritten; re-run only if codebase changes
- `previous_results.jsonl`: each run appends one JSON line; never deleted or modified
- `experiments/claude.md`: updated when merge policy or Fabric rules change; version-stamped

---

## 3. EXPERIMENT_PACKAGE_STRUCTURE

```
experiments/{topic_slug}_v{N}/
├── train.py                   # Fabric entry point — Claude owns
├── module.py                  # TrainingModule (step logic) — Claude owns
├── model.py                   # Architecture — Claude owns, GPT may patch
├── data.py                    # Dataset/DataLoader — Claude owns, GPT may patch
├── configs/
│   ├── default.yaml           # Canonical base config — Claude owns
│   ├── fast.yaml              # Smoke-test override (2 epochs, tiny data)
│   └── ablation/              # One file per ablation variant
│       ├── no_aux_head.yaml
│       └── smaller_backbone.yaml
├── scripts/
│   ├── smoke_test.py          # Forward pass + 2 steps, exit 0/1
│   └── eval.py                # Standalone eval against saved checkpoint
├── tests/
│   └── test_forward.py        # Shape/dtype unit tests — GPT may propose
├── artifacts/
│   ├── checkpoints/           # best.ckpt, last.ckpt
│   ├── logs/                  # TensorBoard / CSV
│   └── metrics/               # per_epoch_metrics.jsonl
├── README.md                  # Auto-generated run summary
└── claude.md                  # Per-experiment spec (copied from experiments/claude.md)
```

**Versioning rule:**
- Each new experiment is a new directory `_v{N}` — never overwrite a prior version
- `_v{N}/configs/` inherit from `_v{N-1}/configs/` unless explicitly changed
- Ablation runs share the same `_v{N}/` package; differentiated by config file only

---

## 4. MODEL_ROLE_SPLIT

### Claude — Architect & Final Integrator
- Generates experiment_spec.json
- Writes initial train.py, module.py, model.py, data.py, configs/default.yaml
- Reviews all GPT patches and Gemini reviews
- Performs final merge into the package
- Makes revision decisions (Path A/B/C)
- Owns experiments/claude.md

### GPT/Codex — Code Patcher & Test Writer
- Proposes patches to model.py, data.py, tests/, configs/
- Format: unified diff saved to `proposals/gpt_patch_{timestamp}.diff`
- Also proposes `tests/test_forward.py` content
- Must not write to train.py, module.py, or hypothesis.json
- Each patch must include: `target_file`, `rationale`, `hypothesis_alignment_check`

### Gemini — Result Interpreter & Design Critic
- Reads result_summary.json and proposes revision direction
- Critiques experiment design (baselines, ablations, controls)
- Format: `proposals/gemini_review_{timestamp}.json`
- Must not write code; outputs structured JSON review only
- Required fields: `verdict`, `issues[]`, `suggested_path` (A/B/C), `evidence_strength`

**No-direct-write rule:** GPT and Gemini write only to `proposals/`. Claude reads proposals and applies what aligns with the hypothesis. Proposal files are archived, never deleted.

---

## 5. MERGE_POLICY

**Priority order (highest to lowest):**
1. Hypothesis alignment — change must serve the stated hypothesis mechanism
2. Metric compatibility — must not break comparability with previous runs
3. Experiment runnability — merged code must pass smoke_test.py
4. Minimal change — prefer the smallest patch that addresses the issue
5. No speculative complexity — reject features not required by experiment_plan

**GPT patch acceptance criteria:**
- [ ] patch applies cleanly to target file
- [ ] smoke_test.py passes after patch
- [ ] `hypothesis_alignment_check` field confirms alignment
- [ ] no new external dependencies unless approved
- [ ] no verbatim copying of >20 consecutive lines from public repos (checked by Claude review)

**Gemini review acceptance criteria:**
- [ ] `verdict` is one of: `accept_as_is | accept_with_patch | revise_experiment | refine_hypothesis | replace_hypothesis`
- [ ] `evidence_strength` is quantified (low/medium/high + justification)
- [ ] `suggested_path` matches evidence_strength threshold

**Conflict resolution:** Claude's judgment is final. When GPT and Gemini disagree, Claude selects the option with stronger hypothesis alignment and weaker complexity increase.

---

## 6. GITLAB_EXECUTION_DESIGN

```yaml
# .gitlab-ci.yml — see template file
```

**Stage sequence:**
1. `setup`: conda/venv, pip install, validate experiment package structure
2. `smoke`: `python scripts/smoke_test.py` — must exit 0 within 3 minutes
3. `train`: `python train.py --config configs/default.yaml` — captures stdout
4. `package`: parse METRICS line, save artifacts, write result_summary.json

**METRICS contract (mandatory):**
```
METRICS:{"psnr": 28.5, "ssim": 0.82, "params_M": 3.1, "inference_ms": 12.4}
```
- Must be exactly one line matching `^METRICS:\{.*\}$`
- train.py is responsible for printing this line at the very end
- If absent or malformed: job fails with `metrics_parse_error`

**Failure capture:**
| Event | Action |
|---|---|
| smoke fail | abort, write `status: smoke_failed` to result_summary |
| train crash | capture stderr last 200 lines, write `status: crashed` |
| timeout (>6h) | GitLab kills job, runner writes `status: timeout` |
| METRICS absent | write `status: metrics_parse_error`, save last 50 stdout lines |
| OOM | captured via stderr, write `status: oom_error` |

**Artifacts saved (always, even on failure):**
- `artifacts/checkpoints/last.ckpt` (if exists)
- `artifacts/logs/` (all)
- `artifacts/metrics/per_epoch_metrics.jsonl`
- `result_summary.json`
- `gitlab_run_metadata.json` (job_id, runner, duration, git_sha)

---

## 7. RESULT_SUMMARY_SCHEMA

See `schemas/result_summary.json` for full JSON Schema.

**Key fields:**
```json
{
  "schema_version": "1.0",
  "run_id": "{topic_slug}_v{N}_{YYYYMMDD_HHMMSS}",
  "hypothesis_id": "...",
  "experiment_version": "N",
  "status": "success | failed | timeout | smoke_failed | oom_error | metrics_parse_error",
  "primary_metric": {"name": "psnr", "value": 28.5, "unit": "dB", "target": 30.0, "met": false},
  "secondary_metrics": [...],
  "deltas_vs_baseline": {"psnr_delta": -1.5, "baseline_run_id": "..."},
  "training_stability": {"loss_converged": true, "nan_detected": false, "lr_schedule_ok": true},
  "bottleneck_candidates": ["small_dataset", "lr_too_high"],
  "ablation_findings": [],
  "confidence": 0.7,
  "recommended_next_actions": [{"path": "A", "rationale": "...", "priority": "high"}],
  "gitlab_run_metadata": {"job_id": "...", "duration_s": 3600, "git_sha": "..."}
}
```

---

## 8. REVISION_POLICY

### Path A — Experiment/Code Fix (Hypothesis Stable)

**Triggering conditions:**
- Primary metric < target but loss converged (training OK, model underpowered)
- NaN/divergence detected (unstable training)
- Smoke test failed (broken code)
- Baselines missing or unfair comparison
- Ablations absent when required by experiment_plan

**Evidence threshold:** 1 run is sufficient for code bugs; 2+ runs for hyperparameter issues

**Confidence requirement:** low (implementation problems are clear-cut)

**May change:** experiment_plan.json, model.py, data.py, configs/, train.py, module.py

**Must remain stable:** hypothesis.json, hypothesis_id, approval.json

**Comparability:** new version inherits configs/_v{N-1}/; baseline_run_id must point to last stable run

**Summary record written:** `revision_request.json` with `path: "A"`, diff of changed files

---

### Path B — Hypothesis Refinement (Core Idea Valid)

**Triggering conditions:**
- Primary metric close to target (within 10%) but claim needs narrowing
- Result is condition-dependent (e.g., works on high-noise but not low-noise)
- Core mechanism confirmed but scope over-claimed
- Gemini review returns `evidence_strength: medium` with `suggested_path: B`

**Evidence threshold:** 2+ runs with consistent pattern

**Confidence requirement:** medium (pattern must be reproducible)

**May change:** hypothesis.json (add `refined_from`, narrow scope), experiment_plan.json

**Must remain stable:** hypothesis core mechanism, hypothesis_id (add `_r{N}` suffix)

**Comparability:** previous runs remain valid as lower-bound baselines

**Summary record written:** `revision_request.json` with `path: "B"`, old/new hypothesis diff

---

### Path C — Hypothesis Replacement

**Triggering conditions (ALL must be met):**
- 3+ runs with strong controls all fail to reach target
- Clear falsification: result contradicts the stated mechanism (not just low metrics)
- Gemini review returns `evidence_strength: high` with `suggested_path: C`
- Claude confirms no Path A/B fix is plausible

**Evidence threshold:** ≥3 independent runs; must include ablation ruling out implementation issues

**Confidence requirement:** high (irreversible — requires explicit justification)

**May change:** hypothesis.json (new `hypothesis_id`, `supersedes` links old), experiment_plan.json, model.py

**Must remain stable:** approval.json, previous_results.jsonl (append-only), all prior artifacts

**Comparability:** old runs are archived as `negative_evidence`; new baseline set after first successful Path C run

**Summary record written:** `revision_request.json` with `path: "C"`, full falsification evidence

---

**Decision rule (enforced):** Prefer Path A → Path B → Path C. Do not escalate without meeting the evidence threshold. A single failed run never triggers Path C.

---

## 9. CLAUDE_MD_SPEC

See `experiments/claude.md` for the full spec.

**Required sections:**
1. Source priority (hypothesis > experiment_plan > code_analysis > prior results)
2. File layout rules (what Claude writes, what GPT may patch, what Gemini cannot touch)
3. Fabric coding rules (init location, wrap order, seeding, precision)
4. Output contract rules (METRICS line format, epoch logging format)
5. Revision rules (Path A/B/C triggers and constraints)
6. Anti-drift rules (no undeclared dependencies, no speculative features)
7. Public-code reuse rules (max 20 consecutive lines, always cite source)
8. Comparability rules (config inheritance, metric naming freeze)
9. Metric naming rules (canonical names, units)
10. Patch/update discipline (proposal format, review checklist)

---

## 10. PYTORCH_FABRIC_RULES

See `experiments/claude.md` §Fabric Rules for full specification.

**Summary:**
- Fabric initialized once in `train.py` main block; never in module.py or model.py
- Wrap order: `fabric.setup(model, optimizer)` then `fabric.setup_dataloaders(train_loader, val_loader)`
- Seeding: `fabric.seed_everything(config.seed)` before any data or model instantiation
- Default precision: `bf16-mixed` on CUDA; `32-true` fallback
- Checkpointing: `fabric.save(path, {"model": model, "optimizer": optimizer, "epoch": epoch, "config": config})`
- Logging: one CSV row per epoch to `artifacts/logs/metrics.csv`; final stdout `METRICS:{...json...}`
- Config-driven: all hyperparameters in `configs/default.yaml`; no magic numbers in train.py
- Fallback: if `len(dataset) < batch_size * 10`, log warning and use `drop_last=False`

---

## 11. IMPLEMENTATION_PLAN

### Phase 1 — Schema & Spec (now)
- [ ] Create `experiments/claude.md` (full spec)
- [ ] Create `schemas/experiment_spec.json`
- [ ] Create `schemas/result_summary.json`
- [ ] Create `schemas/revision_request.json`
- [ ] Create `.gitlab-ci.yml` template
- [ ] Create `experiments/template/` package skeleton
- [ ] Create `docs/merge_checklist.md`

### Phase 2 — GitLab Integration
- [ ] Register GitLab runner with GPU
- [ ] Test `.gitlab-ci.yml` with smoke-only run
- [ ] Verify METRICS parsing works end-to-end
- [ ] Verify artifact upload and result_summary.json generation

### Phase 3 — Multi-Model Proposal Workflow
- [ ] Add `proposals/` directory and archiving convention
- [ ] Integrate GPT patch generation into `lab/model_generator.py`
- [ ] Integrate Gemini review generation into `lab/research_loop.py`
- [ ] Add Claude merge step in research_loop

### Phase 4 — Revision Loop Automation
- [ ] Implement Path A/B/C decision logic in `lab/research_loop.py`
- [ ] Wire revision_request.json → experiment re-generation
- [ ] Add `previous_results.jsonl` append logic
- [ ] Test full loop: run → fail → Path A → re-run → pass

---

## 12. RISKS_AND_GUARDRAILS

| Risk | Guardrail |
|---|---|
| **Model drift** (GPT/Gemini gradually overwrite core logic) | No-direct-write rule; all changes go through proposals/; Claude reviews every merge |
| **Incompatible metric names** | Canonical metric registry in claude.md §Metric Naming; metric names frozen after v1 |
| **Broken comparability** | Config inheritance rule; baseline_run_id required in all result_summary.json |
| **Overfitting to one failed run** | Path C requires ≥3 runs + strong controls; single failure always starts with Path A |
| **Uncontrolled code complexity** | Minimal-change merge policy; GPT patches must pass `complexity_delta ≤ 50 LOC` check |
| **Public-code over-copying** | Max 20 consecutive lines rule; Claude explicitly checks before merging GPT patches |
| **Brittle automation contracts** | METRICS stdout contract is immutable; schema_version field in all JSON artifacts |
| **GitLab runtime instability** | Smoke test stage catches environment issues before 6h train; timeout → Path A auto-trigger |
