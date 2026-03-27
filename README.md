# NTS AI Data Assistant via Autotask API

This demo application showcases how the NTS team can leverage AI models to automatically generate Python code, execute it, and handle errors in real time.

- Project Name: Autotask chat
- Team Members: Shawn Tan
- Description: An AI-powered chat application for Autotask that allows users to retrieve and analyze tickets using natural language conversations.
- What problem does it solve: Enables users to query and analyze data using plain language, eliminating the need for Python programming knowledge.
- Tech stack: OpenAI, Python, Websocket, FastAPI, Autotask ATWS, Caddy Security
- Database / APIs used: Autotask
- Demo link: https://autotask.ai.networkthinking.team
- Repository: https://github.com/networkthinking/nts_autotask_chat

## Run

```bash
./venv/bin/python run.py
```

Or specify the bind address explicitly:

```bash
./venv/bin/python run.py 0.0.0.0:8000
```

Then open `http://localhost:8000`.
