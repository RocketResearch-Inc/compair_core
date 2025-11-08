# Compair Core User Guide

This guide walks a new Compair Core user through the end-to-end experience: configuring the runtime, creating an account, collaborating in groups, working with documents, requesting automated feedback, and managing supporting workflows. Examples assume a FastAPI server running locally at `http://127.0.0.1:8000` and use the `requests` library in Python.

## 0. Initial Setup & Configuration

Before hitting the API, configure the deployment with environment variables. The following values cover the common scenarios for a local or self-hosted deployment:

| Variable | Purpose | Typical local value |
| --- | --- | --- |
| `COMPAIR_EDITION` | Keeps the code in "core" mode. | `core` |
| `COMPAIR_REQUIRE_AUTHENTICATION` | Enables full sign-up/login flows. Disable for single-user demos. | `true` (or `false` for kiosks) |
| `COMPAIR_REQUIRE_EMAIL_VERIFICATION` | When auth is enabled, decide if users must confirm via email before activation. | `false` |
| `COMPAIR_DATABASE_URL` | Points to a custom PostgreSQL/MySQL instance instead of SQLite. | Leave unset to use SQLite |
| `COMPAIR_DB_DIR` / `COMPAIR_DB_NAME` | Location of the SQLite database file when `COMPAIR_DATABASE_URL` is omitted. | `${HOME}/.compair-core/data` / `compair.db` |
| `COMPAIR_VECTOR_BACKEND` | Chooses how embeddings are stored (`json` for SQLite, `pgvector` for PostgreSQL). | `json` |
| `COMPAIR_GENERATION_PROVIDER` | Selects the feedback generator: `local`, `openai`, `http`, or `fallback`. | `local` |
| `COMPAIR_LOCAL_MODEL_URL` | Base URL of the bundled/local text generation + embedding service used when `COMPAIR_GENERATION_PROVIDER=local`. | `http://127.0.0.1:9000` |
| `COMPAIR_OCR_ENDPOINT` | HTTP endpoint used for OCR uploads. When set, OCR auto-enables for core deployments. | `http://127.0.0.1:9001/ocr-file` |
| `COMPAIR_EMAIL_BACKEND` | Controls how verification/reset emails are sent. | `console` (logs to stdout) |

> 💡 Export these variables in your shell or drop them into a `.env` file loaded by your process manager (e.g., `uvicorn`, `docker-compose`, `systemd`).

When both `COMPAIR_REQUIRE_AUTHENTICATION=true` and `COMPAIR_REQUIRE_EMAIL_VERIFICATION=true`, configure SMTP credentials via `EMAIL_HOST`, `EMAIL_USER`, and `EMAIL_PW` so the server can send verification and reset emails. Leave verification disabled (the default) to avoid the mailer dependency during quick tests.

When `COMPAIR_OCR_ENDPOINT` is defined (for example, pointing at the bundled Tesseract service on port `9001`), the server automatically reports OCR capability and allows `/upload/ocr-file`. If the endpoint is missing or unreachable, OCR stays disabled and the API returns `501 Not Implemented`.

Install optional extras as needed: `pip install "compair-core[ocr]"` adds Pillow, pytesseract, and pypdf for in-process OCR support. When running outside the container, install the Tesseract CLI separately (e.g., `brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Debian/Ubuntu) so pytesseract can find the binary.

### Local embeddings and generation

Compair Core ships with helper services that let you run everything offline:

1. **Local embeddings + feedback (default)** – When `COMPAIR_GENERATION_PROVIDER=local`, the server calls `COMPAIR_LOCAL_MODEL_URL` (defaults to `http://127.0.0.1:9000`) to create embeddings and templated feedback. The bundled FastAPI worker uses deterministic hash embeddings and a heuristic feedback stub so you can run 100% offline without extra dependencies. Override the service with your own HTTP endpoint if you need higher-fidelity models.
2. **OpenAI embeddings** – Set `COMPAIR_EMBEDDING_PROVIDER=openai`, supply `COMPAIR_OPENAI_API_KEY`, and optionally override `COMPAIR_OPENAI_EMBED_MODEL` (defaults to `text-embedding-3-small`) to call the OpenAI Embeddings API while still using the local feedback provider.
3. **Custom models** – Swap in your own container or script that exposes the same HTTP interface. Ensure the embedding dimensionality matches `COMPAIR_EMBEDDING_DIM` (defaults to 384). If you change the embedding size, update existing tables or recreate the database to avoid shape mismatches.
4. **OpenAI or other feedback providers** – To call OpenAI for feedback, set `COMPAIR_GENERATION_PROVIDER=openai`, provide `COMPAIR_OPENAI_API_KEY`, and optionally override `COMPAIR_OPENAI_MODEL` (defaults to `gpt-5-nano`). For a bespoke hosted model, use `COMPAIR_GENERATION_PROVIDER=http` and point `COMPAIR_GENERATION_ENDPOINT` at your service.

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

Call `POST /sign-up` with the user’s email (used as `username`), display name, password, and optional group IDs or referral code. The backend validates the email, creates the user record, assigns a private group, triggers analytics, and (when verification is enabled) emails a verification link containing a time-limited token. When `COMPAIR_REQUIRE_EMAIL_VERIFICATION=false`, the account is activated immediately after sign-up.

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

When `COMPAIR_REQUIRE_EMAIL_VERIFICATION=true`, follow the verification link emailed by Compair, which hits `GET /verify-email?token=...`. Verification activates the account, starts a 30-day trial (when trials are enabled), clears the token, and sends pending group invitations if any. Skip this step when verification is disabled.

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

Submitting `POST /join_group` with the target group ID automatically accepts pending invitations, instantly joins public groups, or files a join request for internal groups. Private groups reject direct joins without an invite. On a fresh Compair Core install there are no shared groups yet, so skip ahead to the next section to create one before returning to this join workflow.

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

```python
group_resp = requests.post(
    f"{BASE_URL}/create_group",
    headers=headers,
    data={
        "name": "Product Feedback Guild",
        "category": "Research",
        "description": "Share launch assets and critique drafts.",
        "visibility": "private",
    },
    timeout=10,
).json()
MY_GROUP_ID = group_resp["group_id"]
print("Created group:", MY_GROUP_ID)
```

Group admins can review pending join requests (`GET /admin/join_requests`), approve them (`POST /admin/approve_request`), and list the groups they manage (`GET /admin/groups`). Keep the new `MY_GROUP_ID` handy—the following document examples scope access and feedback to this group.

## 4. Managing Documents

### 4.1 Browse and fetch documents

`GET /load_documents` lists documents you own or, when `own_documents_only=false`, those available through shared/public groups. Filters include publication status and recent activity. Individual documents can be retrieved by title (`GET /load_document`) or ID (`GET /load_document_by_id`), with access checks ensuring you belong to a group that can view them or you authored them.

### 4.2 Create, update, and publish

Create new documents via `POST /create_doc` using form fields for title, type, content, and group assignments. The server enforces per-plan document limits, assigns default groups when none are provided, logs creation activity, and optionally publishes immediately if `is_published=true`.

Update metadata with `POST /update_doc` (same `auth-token` header) and publish/unpublish using `GET /publish_doc?doc_id=...&is_published=true|false`. Suspended users cannot publish until reactivated. Delete single or multiple documents through `GET /delete_doc` or `GET /delete_docs`, which also log the deletion event.

The feedback pipeline only considers **published** documents that share at least one group with the draft being processed. The snippet below creates a reference document inside `MY_GROUP_ID`, publishes it, and processes it once so embeddings are ready for future drafts.

```python
reference_text = (
    "Launch Playbook v1:\n"
    "- Confirm positioning with product and sales.\n"
    "- Publish the launch blog and partner toolkit.\n"
    "- Host a customer webinar within 7 days of GA."
)

reference = requests.post(
    f"{BASE_URL}/create_doc",
    headers=headers,
    data={
        "document_title": "Launch Playbook",
        "document_type": "txt",
        "document_content": reference_text,
        "groups": MY_GROUP_ID,
        "is_published": True,
    },
    timeout=10,
).json()
REFERENCE_DOC_ID = reference["document_id"]

# Process once without generating feedback so the published doc
# has embeddings available for future similarity searches.
requests.post(
    f"{BASE_URL}/process_doc",
    headers=headers,
    data={
        "doc_id": REFERENCE_DOC_ID,
        "doc_text": reference_text,
        "generate_feedback": False,
    },
    timeout=10,
)
```

Create a draft document in the same group whenever you want feedback that leverages the published reference:

```python
candidate_text = (
    "Team,\n"
    "We're announcing Aurora Analytics next month. "
    "Product already approved the messaging, but we still need the partner toolkit "
    "and webinar deck before GA."
)

candidate = requests.post(
    f"{BASE_URL}/create_doc",
    headers=headers,
    data={
        "document_title": "Aurora Launch Email",
        "document_type": "txt",
        "document_content": candidate_text,
        "groups": MY_GROUP_ID,
        "is_published": False,
    },
    timeout=10,
).json()
CANDIDATE_DOC_ID = candidate["document_id"]
```

## 5. Generating and Reviewing Automated Feedback

### 5.1 Trigger document processing

Send updated text to `POST /process_doc` to chunk the document, generate embeddings, and (optionally) request AI feedback. Only the author may process a document; suspended users can save edits but do not receive new feedback. Cloud deployments queue work asynchronously (returning a `task_id`), while core deployments process immediately and return `None`. Because the reference document is already published and processed, the draft below can pick it up as a candidate reference.

Check asynchronous progress with `GET /status/{task_id}` when a task ID is provided.

```python
process_resp = requests.post(
    f"{BASE_URL}/process_doc",
    headers=headers,
    data={
        "doc_id": CANDIDATE_DOC_ID,
        "doc_text": candidate_text,
        "generate_feedback": True,
    },
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
    f"{BASE_URL}/documents/{CANDIDATE_DOC_ID}/feedback",
    headers=headers,
    timeout=10,
)
for fb in feedback_resp.json()["feedback"]:
    print(f"{fb['feedback_id']}: {fb['feedback']}")
```

## 6. Optional: OCR Imports

If OCR is enabled, upload files to convert them into text with `POST /upload/ocr-file`. The endpoint forwards the binary to the configured OCR provider and returns a task ID whose status can be polled with `GET /ocr-file-result/{task_id}`. Editions without OCR support return a 501 error.

The bundled OCR service accepts PDFs (text extraction via pypdf) and common image types such as PNG or JPEG (direct pytesseract). Other formats fall back to plain-text decoding when `COMPAIR_LOCAL_OCR_FALLBACK=text`.

```python
with open("scan.pdf", "rb") as handle:
    ocr_resp = requests.post(
        f"{BASE_URL}/upload/ocr-file",
        headers=headers,
        files={"file": ("scan.pdf", handle, "application/pdf")},
        timeout=20,
    )
ocr_task = ocr_resp.json()
task_id = ocr_task["task_id"]

result_resp = requests.get(
    f"{BASE_URL}/ocr-file-result/{task_id}",
    headers=headers,
    timeout=10,
).json()
print("OCR status:", result_resp.get("status"))
print("OCR text:", result_resp.get("extracted_text", "")[:500])
```

---

## 7. Additional configuration references

Most deployments do not need the knobs below, but they are available for advanced setups:

- `COMPAIR_SINGLE_USER_USERNAME` / `COMPAIR_SINGLE_USER_NAME` – Customize the auto-provisioned account when authentication is disabled.
- `COMPAIR_INCLUDE_LEGACY_ROUTES` – Opt in to the larger API surface that matches the hosted product.
- `COMPAIR_EMBEDDING_DIM` – Override the embedding vector length stored in the database. Changing this requires a matching model and may necessitate reinitializing embeddings.
- `COMPAIR_EMBEDDING_PROVIDER` – Choose `local` (default) or `openai` for embeddings without affecting the feedback provider.
- `COMPAIR_OPENAI_EMBED_MODEL` – Embedding model name when `COMPAIR_EMBEDDING_PROVIDER=openai` (defaults to `text-embedding-3-small`).
- `COMPAIR_REQUIRE_EMAIL_VERIFICATION` – Require email confirmation before activating users (defaults to `false` for core demos).
- `COMPAIR_OPENAI_API_KEY` / `COMPAIR_OPENAI_MODEL` – Required when using the OpenAI generator.
- `COMPAIR_OPENAI_REASONING_EFFORT` – Override the `reasoning.effort` value (`low`, `medium`, or `high`) sent to OpenAI reasoning models.
- `COMPAIR_GENERATION_ENDPOINT` – Target for custom HTTP-based generation services.
- `COMPAIR_OCR_ENDPOINT` – Override the OCR endpoint when connecting to an external OCR processor. Defaults to the bundled service at `http://127.0.0.1:9001/ocr-file` when running the container image.
- `COMPAIR_OCR_REQUEST_TIMEOUT` – Adjust the HTTP timeout (seconds) used when calling the OCR endpoint (default: `30`).
- `EMAIL_HOST` / `EMAIL_USER` / `EMAIL_PW` – SMTP credentials used when sending verification or password reset emails.

Consult `compair_core/server/settings.py` for the authoritative list and defaults.
