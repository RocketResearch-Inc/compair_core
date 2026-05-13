# Compair Core

Compair Core is the self-hosted runtime for Compair’s cross-repo review workflow. It lets you evaluate Compair locally, run the API inside your own environment, and choose whether feedback comes from local providers, OpenAI, or an OpenAI-compatible endpoint.

Compair is built for teams that maintain related codebases: backend plus frontend, API plus SDK, CLI plus cloud service, docs plus implementation, or internal tools split across several repos. Core gives those repos a shared review context so Compair can surface drift, hidden overlap, and missing downstream updates before they become user-facing problems.

If you are starting from a terminal, install the [Compair CLI](https://github.com/RocketResearch-Inc/compair-cli) first:

| Platform | Fast install path |
| --- | --- |
| macOS | `brew tap RocketResearch-Inc/tap` then `brew install --cask compair` |
| Debian / Ubuntu | `curl -fsSL https://rocketresearch-inc.github.io/compair-packages/install/debian.sh \| bash` |
| Fedora / RHEL | `curl -fsSL https://rocketresearch-inc.github.io/compair-packages/install/compair.repo \| sudo tee /etc/yum.repos.d/compair.repo >/dev/null` then `sudo dnf install -y compair` |
| Windows | Download the latest CLI zip from [GitHub Releases](https://github.com/RocketResearch-Inc/compair-cli/releases). |

If you want the fastest first look, start with the [CLI’s](https://github.com/RocketResearch-Inc/compair-cli) offline demo:

```bash
compair demo --offline
```

If you want a real local review against self-hosted Core:

```bash
compair core up
compair profile use local
compair login
compair demo --mode local
```

For stronger local review quality, keep embeddings local and bring your own OpenAI key for generation:

```bash
export OPENAI_API_KEY="sk-..."
compair core config set --generation-provider openai --embedding-provider local --openai-model gpt-5.4-mini --openai-api-key "$OPENAI_API_KEY"
compair core restart
```

If you do not want the key saved in `~/.compair/core_runtime.yaml`, set `COMPAIR_OPENAI_API_KEY` or `OPENAI_API_KEY` in your shell and omit `--openai-api-key`.

If you want Compair Cloud instead of self-hosted Core:

```bash
compair profile use cloud
compair signup --email you@example.com --name "Your Name"
compair login
compair demo --mode cloud
```

Skip `compair signup` if you already have an account.

Use Compair Cloud when you want the strongest out-of-the-box review quality, hosted collaboration, and the least setup. Use Core when you want self-hosting, private evaluation, local development, or a bring-your-own-key path.

## What Core Does

- Runs the Compair API locally with FastAPI.
- Tracks repo snapshots and changed chunks for cross-repo review.
- Generates feedback with local, OpenAI, OpenAI-compatible HTTP, or fallback providers.
- Scores feedback into notification-style events so reports and gates can prioritize likely conflicts.
- Supports a default single-user mode for quick local use, plus account-style auth when you need it.
- Provides the base package used by the published Core container and the hosted Cloud deployment.

## Choose a Review Quality Path

Core is functional in fully local mode, but review quality depends on the provider path you choose.

- Use `compair demo --offline` to understand the workflow with no setup.
- Use local Core with bundled providers for private/offline smoke tests.
- Use local Core with your own OpenAI key for stronger review quality.
- Use Cloud for the strongest hosted/team-ready experience.

| Path | Best for | Notes |
| --- | --- | --- |
| Core + local providers | Offline/private smoke tests | Functional and zero external API cost, but lower-fidelity feedback. |
| Core + local embeddings + OpenAI generation | Cost-aware self-hosted review | Recommended bring-your-own-key starting point. |
| Core + OpenAI embeddings + OpenAI generation | Highest current self-hosted quality | Closest Core lane to the hosted review experience. |
| Compair Cloud | Fastest team-ready experience | Hosted auth, shared accounts, delivery, and the strongest default review quality. |

## Run Core Manually

Most users should start with the CLI-managed Core runtime above. If you want to run the published container image yourself, use:

```bash
docker run -d --name compair-core \
  -p 8000:8000 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  compairsteven/compair-core
```

To use your own OpenAI credentials instead of the bundled local model runtime, the recommended starting point is OpenAI generation with local embeddings:

```bash
export OPENAI_API_KEY="sk-..."
compair core config set --generation-provider openai --embedding-provider local --openai-model gpt-5.4-mini --openai-api-key "$OPENAI_API_KEY"
compair core up
```

If you want the strongest current self-hosted quality path, move both generation and embeddings to OpenAI:

```bash
compair core config set --provider openai --openai-model gpt-5.4 --openai-api-key "$OPENAI_API_KEY"
compair core up
```

Manual equivalent for the lower-outsourced-cost path:

```bash
docker run -d --name compair-core-openai \
  -p 8000:8000 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  -e COMPAIR_GENERATION_PROVIDER=openai \
  -e COMPAIR_EMBEDDING_PROVIDER=local \
  -e COMPAIR_OPENAI_API_KEY="$COMPAIR_OPENAI_API_KEY" \
  -e COMPAIR_OPENAI_MODEL=gpt-5.4-mini \
  -e COMPAIR_OPENAI_EMBED_MODEL=text-embedding-3-small \
  compairsteven/compair-core
```

This path requires only your own OpenAI API key from the Compair side. OpenAI usage is billed by OpenAI.
Keeping embeddings local is the better cost-aware default; using OpenAI for both generation and embeddings is the quality-first option when you want the closest local behavior to Cloud.

For a fuller self-hosted walkthrough, see `docs/quickstart.md` and `docs/user-guide.md`.

## Install as a Python Package

Most people evaluating Compair should start with the CLI-managed Core runtime above. Use the Python package when you want to embed Core in another FastAPI app, develop against the internals, or reuse the shared models and utilities.

```bash
pip install compair-core
```

This installs the package as a dependency so you can embed Compair into your own FastAPI instance or reuse the models in scripts.

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

The optional OCR stack relies on the Tesseract CLI. When running outside the container image, install Tesseract separately, for example `brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Debian/Ubuntu.

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
- `COMPAIR_OPENAI_API_KEY` / `COMPAIR_OPENAI_MODEL` – when using the OpenAI provider, supply your API key and optional model name (defaults to `gpt-5.4-mini`). The fallback kicks in automatically if the key or SDK is unavailable.
- `COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD` / `COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD` – optional pricing hints for `compair review --now`. Set these if you want the CLI quote, markdown report, and backend logs to include an estimated per-run cost, including for OpenAI-compatible self-hosted models.
- `COMPAIR_NOW_REVIEW_MAX_OUTPUT_TOKENS` (`2200`) – maximum output-token budget used for `compair review --now` quotes and model calls.
- Cloud-only `review --now` credit settings: `COMPAIR_REVIEW_NOW_CREDIT_PRICE_ID` configures the Stripe price used for prepaid credit checkout, `COMPAIR_REVIEW_NOW_CREDIT_PACK_CENTS` sets the credit pack value, `COMPAIR_REVIEW_NOW_MAX_QUOTE_CENTS` caps a single quoted run, `COMPAIR_REVIEW_NOW_MIN_CHARGE_CENTS` sets the minimum nonzero charge, and `COMPAIR_REVIEW_NOW_QUOTE_TTL_SEC` controls quote expiry.
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

## Repository Layout

| Path | Purpose |
| --- | --- |
| `compair/` | Core runtime (ORM models, tasks, embeddings, feedback). |
| `server/` | FastAPI app factory and dependency providers used by both editions. |
| `compair_email/` | Console mailer + minimal templates for account verification and password reset. |
| `docs/` | Additional documentation for running and evaluating Core. |

## Core vs. Cloud

Core and Cloud share the same document, group, feedback, and authentication foundations, but they do not expose the same product surface.

- `compair_core` is the self-hosted MIT-licensed runtime.
- The hosted Cloud offering adds the hosted-only layers: Google OAuth, billing, richer analytics, and hosted notification delivery.

Core now includes ranked notification-event generation, `/notification_events`, and `/get_activity_feed` so the CLI, desktop app, and self-hosted evaluations can use the same review semantics as Cloud. Hosted-only delivery layers such as Google OAuth, billing, and transactional notification delivery still belong to the hosted Cloud offering.

Practical guidance:

- Choose **Cloud** when you want the best first-run review quality, hosted collaboration, and the least operational setup.
- Choose **Core** when you want self-hosting, local/private evaluation, or control over your own runtime.
- Choose **Core + your own OpenAI key** when you want self-hosting but still want review quality closer to the hosted experience.

## Tests / Linting

Core currently ships with a syntax sanity check (`python -m compileall ...`). You can add pytest or other tooling as needed.

End-user release and packaging automation live in the `compair-cli` repository, since that repo owns the published binaries and package-manager integrations.

## Reporting Issues

If you try Core, the most helpful feedback is what you ran, what worked, what broke, and whether the review output made sense for your repo shape.

Please open GitHub issues or PRs against this repository with what you tried, what worked, what broke, and where the output was confusing. If you are testing multiple related repos, include the repo shapes involved, for example backend/frontend, API/SDK, CLI/cloud service, or docs/implementation.

Core is MIT licensed. Cloud-specific hosted services remain part of the hosted Compair product.
