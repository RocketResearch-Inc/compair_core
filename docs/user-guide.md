# Compair Core User Guide

This guide walks a new Compair Core user through the end-to-end experience: configuring the runtime, creating an account, collaborating in groups, working with documents, requesting automated feedback, and managing supporting workflows. Examples assume a FastAPI server running locally at `http://127.0.0.1:8000` and use the `requests` library in Python.

## 0. Initial Setup & Configuration

Before hitting the API, configure the deployment with environment variables. The following values cover the common scenarios for a local or self-hosted deployment:

| Variable | Purpose | Typical local value |
| --- | --- | --- |
| `COMPAIR_EDITION` | Keeps the code in "core" mode. | `core` |
| `COMPAIR_REQUIRE_AUTHENTICATION` | Enables full sign-up/login flows. Disable for single-user demos. | `true` (or `false` for kiosks) |
| `COMPAIR_DATABASE_URL` | Points to a custom PostgreSQL/MySQL instance instead of SQLite. | Leave unset to use SQLite |
| `COMPAIR_DB_DIR` / `COMPAIR_DB_NAME` | Location of the SQLite database file when `COMPAIR_DATABASE_URL` is omitted. | `${HOME}/.compair-core/data` / `compair.db` |
| `COMPAIR_VECTOR_BACKEND` | Chooses how embeddings are stored (`json` for SQLite, `pgvector` for PostgreSQL). | `json` |
| `COMPAIR_GENERATION_PROVIDER` | Selects the feedback generator: `local`, `openai`, `http`, or `fallback`. | `local` |
| `COMPAIR_LOCAL_MODEL_URL` | Base URL of the bundled/local text generation + embedding service used when `COMPAIR_GENERATION_PROVIDER=local`. | `http://127.0.0.1:9000` |
| `COMPAIR_OCR_ENDPOINT` | HTTP endpoint used for OCR uploads. Defaults to the bundled Tesseract sidecar. | `http://127.0.0.1:9001/ocr-file` |
| `COMPAIR_EMAIL_BACKEND` | Controls how verification/reset emails are sent. | `console` (logs to stdout) |

> 💡 Export these variables in your shell or drop them into a `.env` file loaded by your process manager (e.g., `uvicorn`, `docker-compose`, `systemd`).

Install optional extras as needed: `pip install "compair-core[ocr]"` adds Pillow and pytesseract for in-process OCR support.

### Local embeddings and generation

Compair Core ships with helper services that let you run everything offline:

1. **Local embeddings** – When `COMPAIR_GENERATION_PROVIDER=local`, the server calls the URL in `COMPAIR_LOCAL_MODEL_URL` to create embeddings and feedback. Point this URL at the FastAPI worker started by `compair_core.compair.local_models`. The helper service defaults to a small sentence-transformer (384 dimensions) and a lightweight instruction-tuned text generator. Both models run on CPU by default.
2. **Custom models** – Swap in your own container or script that exposes the same HTTP interface. Ensure the embedding dimensionality matches `COMPAIR_EMBEDDING_DIM` (defaults to 384). If you change the embedding size, update existing tables or recreate the database to avoid shape mismatches.
3. **OpenAI or other providers** – To call OpenAI, set `COMPAIR_GENERATION_PROVIDER=openai`, provide `COMPAIR_OPENAI_API_KEY`, and optionally override `COMPAIR_OPENAI_MODEL` (defaults to `gpt-4o-mini`). For a bespoke hosted model, use `COMPAIR_GENERATION_PROVIDER=http` and point `COMPAIR_GENERATION_ENDPOINT` at your service.

If you do not run any model service, set `COMPAIR_GENERATION_PROVIDER=fallback` to skip generation while still storing document embeddings (useful for similarity-only scenarios).

## Deploying with Docker

The `compairsteven/compair-core` image bundles the API, local model, and Tesseract-based OCR service. Run it with the ports you plan to expose and the environment you would normally supply to the server:

```bash
docker run -d --name compair-core \
  -p 8000:8000 -p 9000:9000 -p 9001:9001 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  compairsteven/compair-core
```

Inside the container:

- `uvicorn` serves the API on `0.0.0.0:8000`.
- The local model service listens on `0.0.0.0:9000`; `COMPAIR_LOCAL_MODEL_URL` already points to it.
- The OCR sidecar listens on `0.0.0.0:9001/ocr-file`; override `COMPAIR_OCR_ENDPOINT` if you want to point at an external OCR processor.

When running behind a reverse proxy (e.g., Traefik, nginx), forward the ports you intend to expose and mount a persistent volume at `/data` if you keep the default SQLite database.

---

## 1. Onboarding & Access

### 1.1 Create an account

Call `POST /sign-up` with the user’s email (used as `username`), display name, password, and optional group IDs or referral code. The backend validates the email, creates the user record, adds them to the welcome/private groups, triggers analytics, and emails a verification link containing a time-limited token.

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

payload = {
    "username": "new.user@example.com",
    "name": "New User",
    "password": "StrongPass!123",
    "groups": None,
    "referral_code": None,
}
resp = requests.post(f"{BASE_URL}/sign-up", json=payload, timeout=10)
resp.raise_for_status()
print(resp.json())
```

### 1.2 Verify the email address

Follow the verification link emailed by Compair, which hits `GET /verify-email?token=...`. Verification activates the account, starts a 30-day trial (when trials are enabled), clears the token, and sends pending group invitations if any.

### 1.3 Log in and maintain a session

Authenticate with `POST /login`. Successful requests return user metadata and a 24-hour session token (`auth_token`). Provide this token as the `auth-token` header for all subsequent calls that require authentication.

Sessions can be inspected with `GET /load_session?auth_token=...`, which ensures the token is valid and unexpired. When authentication is disabled (single-user mode), the server silently provisions a default user and session.

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

login_resp = requests.post(
    f"{BASE_URL}/login",
    json={"username": "new.user@example.com", "password": "StrongPass!123"},
    timeout=10,
)
login_resp.raise_for_status()
session = login_resp.json()
headers = {"auth-token": session["auth_token"]}
```

### 1.4 Account recovery (optional)

Users can request a reset link with `POST /forgot-password`, then submit the token and new password via `POST /reset-password`. Both endpoints are disabled when authentication is not required.

## 2. Profile & Personal Settings

Update display name, role, group memberships, and personal flags by posting form data to `POST /update_user`. String booleans such as `hide_affiliations` are automatically normalized to real booleans.

```python
update_resp = requests.post(
    f"{BASE_URL}/update_user",
    headers=headers,
    data={
        "name": "New User (Preferred)",
        "role": "Researcher",
        "hide_affiliations": "false",
    },
    timeout=10,
)
update_resp.raise_for_status()
```

## 3. Collaborating Through Groups

### 3.1 Discover and inspect groups

Use `GET /load_groups` to explore public and invited groups, filter by category or visibility, and paginate results. When `user_id` is supplied, the endpoint narrows results to groups the user can access (including invitations).

List members of a specific group with `GET /load_group_users?group_id=...`, which enforces membership/visibility rules before returning paginated user summaries.

### 3.2 Join existing groups

Submitting `POST /join_group` with the target group ID automatically accepts pending invitations, instantly joins public groups, or files a join request for internal groups. Private groups reject direct joins without an invite.

```python
join_resp = requests.post(
    f"{BASE_URL}/join_group",
    headers=headers,
    data={"group_id": "grp_public_123"},
    timeout=10,
)
print(join_resp.json())
```

### 3.3 Create and administer groups

Group creators call `POST /create_group` with name, category, description, visibility, and an optional image file. Internal groups are limited to active team-plan users. The creator automatically becomes an administrator and gains ownership of the new group.

Group admins can review pending join requests (`GET /admin/join_requests`), approve them (`POST /admin/approve_request`), and list the groups they manage (`GET /admin/groups`).

## 4. Managing Documents

### 4.1 Browse and fetch documents

`GET /load_documents` lists documents you own or, when `own_documents_only=false`, those available through shared/public groups. Filters include publication status and recent activity. Individual documents can be retrieved by title (`GET /load_document`) or ID (`GET /load_document_by_id`), with access checks ensuring you belong to a group that can view them or you authored them.

### 4.2 Create, update, and publish

Create new documents via `POST /create_doc` using form fields for title, type, content, and group assignments. The server enforces per-plan document limits, assigns default groups when none are provided, logs creation activity, and optionally publishes immediately if `is_published=true`.

Update metadata with `POST /update_doc` (same `auth-token` header) and publish/unpublish using `GET /publish_doc?doc_id=...&is_published=true|false`. Suspended users cannot publish until reactivated.

Delete single or multiple documents through `GET /delete_doc` or `GET /delete_docs`, which also log the deletion event.

```python
create_resp = requests.post(
    f"{BASE_URL}/create_doc",
    headers=headers,
    data={
        "document_title": "Q1 Market Analysis",
        "document_type": "txt",
        "document_content": "Initial draft content...",
        "is_published": False,
    },
    timeout=10,
)
doc_id = create_resp.json()["document_id"]
```

## 5. Generating and Reviewing Automated Feedback

### 5.1 Trigger document processing

Send updated text to `POST /process_doc` to chunk the document, generate embeddings, and (optionally) request AI feedback. Only the author may process a document; suspended users can save edits but do not receive new feedback. Cloud deployments queue work asynchronously (returning a `task_id`), while core deployments process immediately and return `None`.

Check asynchronous progress with `GET /status/{task_id}` when a task ID is provided.

```python
process_resp = requests.post(
    f"{BASE_URL}/process_doc",
    headers=headers,
    data={"doc_id": doc_id, "doc_text": "Revised content...", "generate_feedback": True},
    timeout=10,
)
print(process_resp.json())
```

### 5.2 Explore chunks, feedback, and references

- Retrieve chunk metadata with `GET /load_chunks?document_id=...`.
- Fetch feedback per chunk using `GET /load_feedback?chunk_id=...`; responses include the system’s text plus any user rating and hide status.
- See all feedback tied to a document via `GET /documents/{document_id}/feedback`, subject to the same access rules as document retrieval.
- Load reference passages drawn from similar documents with `GET /load_references?chunk_id=...`. Each reference includes the document and author details for context.

Users may hide feedback (`POST /feedback/{feedback_id}/hide` with `is_hidden=true|false`) or rate it positively/negatively (`POST /feedback/{feedback_id}/rate`) to provide supervised signals.

```python
feedback_resp = requests.get(
    f"{BASE_URL}/documents/{doc_id}/feedback",
    headers=headers,
    timeout=10,
)
for fb in feedback_resp.json()["feedback"]:
    print(f"{fb['feedback_id']}: {fb['feedback']}")
```

## 6. Optional: OCR Imports

If OCR is enabled, upload files to convert them into text with `POST /upload/ocr-file`. The endpoint forwards the binary to the configured OCR provider and returns a task ID whose status can be polled with `GET /ocr-file-result/{task_id}`. Editions without OCR support return a 501 error.

```python
with open("scan.pdf", "rb") as handle:
    ocr_resp = requests.post(
        f"{BASE_URL}/upload/ocr-file",
        headers=headers,
        files={"file": ("scan.pdf", handle, "application/pdf")},
        timeout=20,
    )
print(ocr_resp.json())
```

---

## 7. Additional configuration references

Most deployments do not need the knobs below, but they are available for advanced setups:

- `COMPAIR_SINGLE_USER_USERNAME` / `COMPAIR_SINGLE_USER_NAME` – Customize the auto-provisioned account when authentication is disabled.
- `COMPAIR_INCLUDE_LEGACY_ROUTES` – Opt in to the larger API surface that matches the hosted product.
- `COMPAIR_EMBEDDING_DIM` – Override the embedding vector length stored in the database. Changing this requires a matching model and may necessitate reinitializing embeddings.
- `COMPAIR_OPENAI_API_KEY` / `COMPAIR_OPENAI_MODEL` – Required when using the OpenAI generator.
- `COMPAIR_GENERATION_ENDPOINT` – Target for custom HTTP-based generation services.
- `COMPAIR_OCR_ENDPOINT` – Override the OCR endpoint when connecting to an external OCR processor. Defaults to the bundled service at `http://127.0.0.1:9001/ocr-file`.

Consult `compair_core/server/settings.py` for the authoritative list and defaults.
