# AGENTS

## Purpose
- Interactive chat website that answers operational questions using the ATWS Python library
- Use websocket
- can use data fetched before to answer follow-up question
- fix errors if generated code has issue
- use `venv/bin/python` and `venv/bin/pip` for python

## Main Structure
- `src/app/main.py`: FastAPI app, websocket flow, cached-context reuse, repair loop.
- `src/app/llm.py`: prompts, reuse decisions, staged code generation, cell repair.
- `src/app/executor.py`: sandboxed execution, result normalization, context previews.
- `src/app/autotask.py`: `atws` helpers, label resolution, picklist helpers, bulk ID lookup helpers.
- `src/static/index.html`: current frontend. It contains the active HTML, CSS, and JS.

## Autotask / atws Rules
- Prefer fetching the smallest dataset possible.
- For ticket company grouping, uses `Ticket.AccountID`.
- For company names, uses `Account.AccountName`.

## Lookup Strategy
- atws and Autotask API is slow. When loop through tickets or timeentries, aggregate first, resolve labels second


## Execution / Repair
- Generated code runs as notebook-like cells.
- Each non-final cell must:
  - validate its output
  - store previewable values in `context`
  - set `context["checkpoint"]` or `context["checkpoint_keys"]`
- Cell failures are repaired and retried.
