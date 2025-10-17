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

## Containers

Container definitions and build pipelines live outside this public package:

- The **core** container lives alongside the private CI workflow in the `compair_cloud` repository (`Dockerfile.core`). It installs this package from PyPI and runs the FastAPI factory with SQLite defaults.
- A **cloud** container (`Dockerfile.cloud`) is built from a private cloud extension that enables premium features. For more information, please visit [https://www.compair.sh/](https://www.compair.sh/).

## Configuration

Key environment variables for the core edition:

- `COMPAIR_EDITION` (`core`) – corresponds to this core local implementation.
- `COMPAIR_SQLITE_DIR` / `COMPAIR_SQLITE_NAME` – override the default local SQLite path (falls back to `./compair_data` if `/data` is not writable).
- `COMPAIR_LOCAL_MODEL_URL` – endpoint for your local embeddings/feedback service (defaults to `http://local-model:9000`).
- `COMPAIR_EMAIL_BACKEND` – the core mailer logs emails to stdout; cloud overrides this with transactional delivery.
- `COMPAIR_REQUIRE_AUTHENTICATION` (`true`) – set to `false` to run the API in single-user mode without login or account management. When disabled, Compair auto-provisions a local user, group, and long-lived session token so you can upload documents immediately.
- `COMPAIR_SINGLE_USER_USERNAME` / `COMPAIR_SINGLE_USER_NAME` – override the email-style username and display name that are used for the auto-provisioned local user in single-user mode.
- `COMPAIR_INCLUDE_LEGACY_ROUTES` (`false`) – opt-in to the full legacy API surface (used by the hosted product) when running the core edition. Leave unset to expose only the streamlined single-user endpoints in Swagger.

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
