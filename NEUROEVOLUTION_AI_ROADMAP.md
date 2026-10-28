# NeuroEvolution AI — architecture and delivery plan

## Goal
Build an auditable research agent that combines a replaceable language model,
an explicitly authorized tool executor, experiment evaluation, and human-approved
promotion of improved models. "Self-improvement" means proposing and measuring
changes, not unrestricted self-modification.

## Components
1. **Model adapter:** interchangeable local/open-weight or hosted LLM; credentials
   provided only at runtime, never committed.
2. **Agent runtime:** bounded reasoning/action steps, tool allowlist, trace, budgets.
3. **Memory:** opt-in retrieval store with provenance, retention and deletion.
4. **Research engine:** isolated experiments, fixed datasets, baselines, seeds,
   environment manifests and artifact hashes.
5. **Evaluator:** held-out tests, regression suites, cost/latency/safety metrics;
   no test-set-driven selection.
6. **Promotion gate:** reviewable diffs and human approval before deployment.

## Phase 1 — safe offline kernel
- [x] Python model interface, bounded tool loop and trace.
- [x] Unit tests for tool allowlisting and budget behavior.
- [x] Verify tests in CI for code commit `a93c7a3` ([run #40632242673](https://github.com/alberto90-code/neuroevolution-lab/actions/runs/40632242673): 38 tests passed, 2026-10-28).
- [ ] Structured JSON protocol and input/output size budgets.
- [ ] Tool error handling, redaction, trace persistence (opt-in).
- [ ] Mock-model benchmark suite with explicit expected results.

## Phase 2 — practical language agent
Connect a local open-weight instruct model through a configurable adapter.
Use sandboxed read-only repository inspection initially. Measure task success,
tool errors, token cost and latency against a fixed benchmark.

## Phase 3 — research automation
Run candidate training configurations in isolated jobs. Compare validation
metrics and resource cost, record dataset hashes and versions, then report
held-out test performance only for the selected candidate.

## Phase 4 — controlled evolution
Allow agents to propose patches in review branches, require passing tests and
human approval to merge. Never auto-expand privileges, leak secrets, change
repository visibility or deploy unreviewed model-generated code.

## Non-goals and constraints
No claim of frontier-level general intelligence; no automatic self-rewriting
in production. Experiments must have budgets and a stop condition. Published
metrics must be reproducible or clearly labeled as unverified.
