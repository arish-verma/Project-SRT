# SRT Agent Constitution

## Source of truth
GitHub repository state and canonical documents are authoritative over chat memory.

## Rules
1. Read `README.md`, `docs/AI_CONTEXT.md`, `docs/ARCHITECTURE.md`, and `docs/STATUS.md` before implementation.
2. Do not silently change architecture, API contracts, event schemas, or model choices.
3. Prefer small, testable modules and explicit interfaces.
4. Never commit secrets, RTSP credentials, private keys, or personal data.
5. Every meaningful change must include tests or an explicit reason why testing is deferred.
6. Do not claim accuracy or capabilities that have not been validated.
7. Keep model implementations replaceable through adapters.
8. Update status/handoff documentation after substantial work.

## Agent ownership
- GPT: architecture, integration, scope and decisions.
- Claude: implementation and integration.
- Gemini: model/CV research and benchmarks.
- Kimi: QA, reliability and acceptance testing.
- Grok: red-team review and technical skepticism.
