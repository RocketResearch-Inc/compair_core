# Compair Core

Compair Core is the open-source foundation of the Compair platform. It bundles the shared data models, FastAPI application, email utilities, and local-only helpers so that you can run Compair in a self-hosted or evaluation environment without premium cloud integrations.

The premium cloud offering (available at [https://www.compair.sh/](https://www.compair.sh/)) layers on premium services (premium models, OCR, storage,  etc.). Core gracefully falls back to local behaviour when those packages are not present.

If you want the strongest out-of-the-box review quality with the least setup, start with Compair Cloud. Core is the self-hosted path: it works well for evaluation and local/private deployments, and it gets closer to Cloud quality when you connect your own OpenAI key instead of relying on the bundled local fallback.

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

## Repository Layout

| Path | Purpose |
| --- | --- |
| `compair/` | Core runtime (ORM models, tasks, embeddings, feedback). |
| `server/` | FastAPI app factory and dependency providers used by both editions. |
| `compair_email/` | Console mailer + minimal templates for account verification and password reset. |
| `docs/` | Additional documentation (see `docs/editions.md` for an overview of the two editions). |

## Containers

Container definitions and build pipelines live outside this public package:

- The **core** container lives alongside the private CI workflow in the `compair_cloud` repository (`Dockerfile.core`). It installs this package from PyPI and runs the FastAPI factory with SQLite defaults.
- A **cloud** container (`Dockerfile.cloud`) is built from a private cloud extension that enables premium features. For more information, please visit [https://www.compair.sh/](https://www.compair.sh/).

If you are evaluating Core locally with the CLI, the simplest path is:

```bash
compair core up
compair profile use local
compair login
```

If you want to run the published container image manually instead, use:

```bash
docker run -d --name compair-core \
  -p 8000:8000 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  compairsteven/compair-core
```

To use your own OpenAI credentials instead of the bundled local model runtime:

```bash
compair core config set --provider openai --openai-api-key "$OPENAI_API_KEY"
compair core up
```

Or manually:

```bash
docker run -d --name compair-core-openai \
  -p 8000:8000 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  -e COMPAIR_GENERATION_PROVIDER=openai \
  -e COMPAIR_EMBEDDING_PROVIDER=openai \
  -e COMPAIR_OPENAI_API_KEY="$COMPAIR_OPENAI_API_KEY" \
  -e COMPAIR_OPENAI_MODEL=gpt-5-nano \
  -e COMPAIR_OPENAI_EMBED_MODEL=text-embedding-3-small \
  compairsteven/compair-core
```

This path requires only your own OpenAI API key from the Compair side. OpenAI usage is billed by OpenAI.

For a fuller self-hosted walkthrough, see `docs/quickstart.md` and `docs/user-guide.md`.

## Configuration

Key environment variables for the core edition:

- `COMPAIR_EDITION` (`core`) – corresponds to this core local implementation.
- `COMPAIR_DATABASE_URL` – optional explicit SQLAlchemy URL (e.g. `postgresql+psycopg2://user:pass@host/db`). When omitted, Compair falls back to a local SQLite file.
- `COMPAIR_DB_DIR` / `COMPAIR_DB_NAME` – directory and filename for the bundled SQLite database (default: `~/.compair-core/data/compair.db`). Legacy `COMPAIR_SQLITE_*` variables remain supported.
- `COMPAIR_LOCAL_MODEL_URL` – endpoint for your local embeddings/feedback service (defaults to `http://127.0.0.1:9000`).
- `COMPAIR_EMBEDDING_PROVIDER` – choose `local` (default) or `openai` for embeddings independent of feedback.
- `COMPAIR_OPENAI_EMBED_MODEL` – override the OpenAI embedding model when `COMPAIR_EMBEDDING_PROVIDER=openai`.
- `COMPAIR_EMAIL_BACKEND` – the core mailer logs emails to stdout; cloud overrides this with transactional delivery.
- `COMPAIR_REQUIRE_AUTHENTICATION` (`false`) – leave this unset for the default single-user Core experience, or set it to `true` to enable full login/account-management flows. When disabled, Compair auto-provisions a local user, group, and long-lived session token so you can upload documents immediately.
- `COMPAIR_REQUIRE_EMAIL_VERIFICATION` (`false`) – require new users to confirm via email before activation. Set to `true` only when SMTP credentials are configured.
- `COMPAIR_SINGLE_USER_USERNAME` / `COMPAIR_SINGLE_USER_NAME` – override the email-style username and display name that are used for the auto-provisioned local user in single-user mode.
- `COMPAIR_INCLUDE_LEGACY_ROUTES` (`false`) – opt-in to the full legacy API surface (used by the hosted product) when running the core edition. Leave unset to expose only the streamlined single-user endpoints in Swagger.
- `COMPAIR_EMBEDDING_DIM` – force the embedding vector size stored in the database (defaults to 384 for core, 1536 for cloud). Keep this in sync with whichever embedding model you configure.
- `COMPAIR_VECTOR_BACKEND` (`auto`) – set to `pgvector` when running against PostgreSQL with the pgvector extension, or `json` to store embeddings as JSON (the default for SQLite deployments).
- `COMPAIR_GENERATION_PROVIDER` (`local`) – choose how feedback is produced. Options: `local` (call the bundled FastAPI service), `openai` (use ChatGPT-compatible APIs with an API key), `http` (POST the request to a custom endpoint), or `fallback` (skip generation and surface similar references only).
- `COMPAIR_OPENAI_API_KEY` / `COMPAIR_OPENAI_MODEL` – when using the OpenAI provider, supply your API key and optional model name (defaults to `gpt-5-nano`). The fallback kicks in automatically if the key or SDK is unavailable.
- `COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD` / `COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD` – optional pricing hints for `compair review --now`. Set these if you want the markdown report and backend logs to include an estimated per-run cost, including for OpenAI-compatible self-hosted models.
- `COMPAIR_GENERATION_ENDPOINT` – HTTP endpoint invoked when `COMPAIR_GENERATION_PROVIDER=http`; the service receives a JSON payload (`document`, `references`, `length_instruction`) and should return `{"feedback": ...}`.
- `COMPAIR_NOTIFICATION_SCORING_ENABLED` (`true`) – enable ranked notification-event scoring in Core. Set to `false` if you only want raw feedback without notification triage.
- `COMPAIR_NOTIFICATION_SCORING_PROVIDER` (`auto`) – choose `auto`, `heuristic`, or `openai` for notification-event scoring. `auto` uses OpenAI when an API key is configured and otherwise falls back to a deterministic local heuristic.
- `COMPAIR_NOTIFICATION_SCORING_TIMEOUT_S` (`30`) – request timeout in seconds for OpenAI-backed notification scoring. Increase this for large cross-repo review runs if scorer requests are timing out.
- `COMPAIR_NOTIFICATION_SCORING_MAX_RETRIES` (`2`) – retry count for OpenAI-backed notification scoring transport failures/timeouts.
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

## Core vs. Cloud

Core and Cloud share the same document, group, feedback, and authentication foundations, but they do not expose the same product surface.

- `compair_core` is the self-hosted/open-core runtime.
- `compair_cloud` adds the hosted-only layers: Google OAuth, billing, richer analytics, and hosted notification delivery.

Core now includes ranked notification-event generation, `/notification_events`, and `/get_activity_feed` so the CLI, desktop app, and self-hosted evaluations can use the same review semantics as Cloud. Hosted-only delivery layers such as Google OAuth, billing, and transactional notification delivery still belong to `compair_cloud`.

Practical guidance:

- Choose **Cloud** when you want the best first-run review quality, hosted collaboration, and the least operational setup.
- Choose **Core** when you want self-hosting, local/private evaluation, or control over your own runtime.
- Choose **Core + your own OpenAI key** when you want self-hosting but still want review quality closer to the hosted experience.

## Tests / Linting

Core currently ships with a syntax sanity check (`python -m compileall ...`). You can add pytest or other tooling as needed.

Release and packaging steps are documented in `docs/maintainers.md`.

## Reporting Issues

Please open GitHub issues or PRs against this repository. If you are a Compair Cloud customer, reach out through your support channel for issues related to premium features.
