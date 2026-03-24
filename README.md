# NTS AI Data Assistant

Interactive web app that:

- accepts plain-English Autotask questions
- uses the configured LLM in `config.yaml`
- generates Python code with `atws`
- runs that code against Autotask
- streams visible reasoning updates to the UI

## Run

```bash
./venv/bin/python run.py
```

Or specify the bind address explicitly:

```bash
./venv/bin/python run.py 0.0.0.0:8000
```

Then open `http://localhost:8000`.
