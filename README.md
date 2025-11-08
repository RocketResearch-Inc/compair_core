# Compair Core

Compair Core is the open-source foundation of the Compair platform. It bundles the shared data models, FastAPI application, email utilities, and local-only helpers so that you can run Compair in a self-hosted or evaluation environment without premium cloud integrations.

The premium cloud offering (available at [https://www.compair.sh/](https://www.compair.sh/)) layers on premium services (premium models, OCR, storage,  etc.). Core gracefully falls back to local behaviour when those packages are not present.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `compair/` | Core runtime (ORM models, tasks, embeddings, feedback). |
| `server/` | FastAPI app factory and dependency providers used by both editions. |
| `compair_email/` | Console mailer + minimal templates for account verification and password reset. |
| `docs/` | Additional documentation (see `docs/editions.md` for an overview of the two editions). |

## Installing

```bash
pip install compair-core
```

This installs the package as a dependency so you can embed Compair into your own FastAPI instance or reuse the models in scripts. The core library exposes hooks for the private cloud extension that Compair itself uses for hosted deployments.

### Installing from source

You can also install directly from GitHub (handy for pinning to a specific commit or branch):

```bash
pip install "git+https://github.com/RocketResearch-Inc/compair_core.git@main"
```

For local development:

```bash
git clone https://github.com/RocketResearch-Inc/compair_core.git
cd compair_core
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> 🔧 The optional OCR stack relies on the Tesseract CLI. When running outside the container image, install Tesseract separately (for example, `brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Debian/Ubuntu) so pytesseract can invoke it.

## Containers

Container definitions and build pipelines live outside this public package:

- The **core** container lives alongside the private CI workflow in the `compair_cloud` repository (`Dockerfile.core`). It installs this package from PyPI and runs the FastAPI factory with SQLite defaults.
- A **cloud** container (`Dockerfile.cloud`) is built from a private cloud extension that enables premium features. For more information, please visit [https://www.compair.sh/](https://www.compair.sh/).

## Configuration

Key environment variables for the core edition:

- `COMPAIR_EDITION` (`core`) – corresponds to this core local implementation.
- `COMPAIR_DATABASE_URL` – optional explicit SQLAlchemy URL (e.g. `postgresql+psycopg2://user:pass@host/db`). When omitted, Compair falls back to a local SQLite file.
- `COMPAIR_DB_DIR` / `COMPAIR_DB_NAME` – directory and filename for the bundled SQLite database (default: `~/.compair-core/data/compair.db`). Legacy `COMPAIR_SQLITE_*` variables remain supported.
- `COMPAIR_LOCAL_MODEL_URL` – endpoint for your local embeddings/feedback service (defaults to `http://127.0.0.1:9000`).
- `COMPAIR_EMBEDDING_PROVIDER` – choose `local` (default) or `openai` for embeddings independent of feedback.
- `COMPAIR_OPENAI_EMBED_MODEL` – override the OpenAI embedding model when `COMPAIR_EMBEDDING_PROVIDER=openai`.
- `COMPAIR_EMAIL_BACKEND` – the core mailer logs emails to stdout; cloud overrides this with transactional delivery.
- `COMPAIR_REQUIRE_AUTHENTICATION` (`true`) – set to `false` to run the API in single-user mode without login or account management. When disabled, Compair auto-provisions a local user, group, and long-lived session token so you can upload documents immediately.
- `COMPAIR_REQUIRE_EMAIL_VERIFICATION` (`false`) – require new users to confirm via email before activation. Set to `true` only when SMTP credentials are configured.
- `COMPAIR_SINGLE_USER_USERNAME` / `COMPAIR_SINGLE_USER_NAME` – override the email-style username and display name that are used for the auto-provisioned local user in single-user mode.
- `COMPAIR_INCLUDE_LEGACY_ROUTES` (`false`) – opt-in to the full legacy API surface (used by the hosted product) when running the core edition. Leave unset to expose only the streamlined single-user endpoints in Swagger.
- `COMPAIR_EMBEDDING_DIM` – force the embedding vector size stored in the database (defaults to 384 for core, 1536 for cloud). Keep this in sync with whichever embedding model you configure.
- `COMPAIR_VECTOR_BACKEND` (`auto`) – set to `pgvector` when running against PostgreSQL with the pgvector extension, or `json` to store embeddings as JSON (the default for SQLite deployments).
- `COMPAIR_GENERATION_PROVIDER` (`local`) – choose how feedback is produced. Options: `local` (call the bundled FastAPI service), `openai` (use ChatGPT-compatible APIs with an API key), `http` (POST the request to a custom endpoint), or `fallback` (skip generation and surface similar references only).
- `COMPAIR_OPENAI_API_KEY` / `COMPAIR_OPENAI_MODEL` – when using the OpenAI provider, supply your API key and optional model name (defaults to `gpt-5-nano`). The fallback kicks in automatically if the key or SDK is unavailable.
- `COMPAIR_GENERATION_ENDPOINT` – HTTP endpoint invoked when `COMPAIR_GENERATION_PROVIDER=http`; the service receives a JSON payload (`document`, `references`, `length_instruction`) and should return `{"feedback": ...}`.
- `COMPAIR_OCR_ENDPOINT` – endpoint the backend calls for OCR uploads. Setting this (e.g., to the bundled Tesseract wrapper at `http://127.0.0.1:9001/ocr-file`) automatically enables OCR.
- `COMPAIR_OCR_REQUEST_TIMEOUT` – timeout in seconds for HTTP OCR requests (default `30`).

When verification is required, configure `EMAIL_HOST`, `EMAIL_USER`, and `EMAIL_PW` so the mailer can deliver verification and password reset emails.

See `compair_core/server/settings.py` for the full settings surface.

## Developing Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn compair_core.server.app:create_app --factory --reload
```

The API will be available at http://127.0.0.1:8000 and supports the Swagger UI at `/docs`.

## Tests / Linting

Core currently ships with a syntax sanity check (`python -m compileall ...`). You can add pytest or other tooling as needed.

Release and packaging steps are documented in `docs/maintainers.md`.

## Reporting Issues

Please open GitHub issues or PRs against this repository. If you are a Compair Cloud customer, reach out through your support channel for issues related to premium features.
