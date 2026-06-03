# Contributing

Thanks for taking the time to help — contributions are genuinely appreciated.

Start with the [`documentation/`](documentation/) hub: it covers the architecture, data model, pipeline, setup, serving, and the [conventions](documentation/guides/conventions.md) every change is held to. Keep pull requests small and focused, and make sure the gates pass before opening one:

```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run lint-imports && uv run python scripts/lint_docs.py && uv run pytest
# frontend: bun run typecheck && bun run lint && bun run test && bun run build
```

## AI-assisted pull requests

Pull requests produced with AI assistance should be generated with at least **Opus 4.8** at **xhigh** effort and code-reviewed by **GPT-5.5** at **xhigh** effort before submission.
