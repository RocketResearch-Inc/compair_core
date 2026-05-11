# Compair Core Quickstart

This quickstart shows the fastest ways to try Compair Core, the self-hosted runtime for Compair's cross-repo review workflow.

Start with the CLI-managed path if you want to test real repo review behavior. Use the manual API walkthrough when you want to embed Core, inspect endpoints, or build against the backend directly.

Compair Core includes local notification-event ranking so self-hosted reviews can use the same report and gating semantics as Cloud. Hosted delivery layers such as Google OAuth, billing, and transactional email remain Cloud-only.

## Zero-setup sample

If you only want to see the output shape before starting Docker or configuring a model key:

```bash
compair demo --offline
```

That command creates the demo repos and renders a prebaked report. It is not a live review, but it is the quickest way to see the problem Compair is built to catch.

## Managed container quick start

If you are trying Compair from the CLI, let the CLI manage the local Core container for you:

```bash
compair core up
compair profile use local
compair login
compair demo --mode local
```

This is the recommended first live Core run.

## Manual container quick start

Prefer to run the prebuilt image yourself? The published container already bundles the API, local model, and OCR sidecar:

```bash
docker run -d --name compair-core \
  -p 8000:8000 \
  -e COMPAIR_REQUIRE_AUTHENTICATION=false \
  compairsteven/compair-core
```

The container exposes:

- `http://localhost:8000` – Compair Core API
- the bundled local model and OCR helpers stay internal to the container unless you choose to expose them separately

You can override the OCR endpoint by setting `COMPAIR_OCR_ENDPOINT` when starting the container.

### Use your own OpenAI key instead of the local model

The recommended self-hosted starting point is OpenAI generation with local embeddings:

```bash
compair core config set --generation-provider openai --embedding-provider local --openai-api-key "$OPENAI_API_KEY"
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

Keeping embeddings local is the better cost-aware default. Use OpenAI for both generation and embeddings when you want the quality-first self-hosted path instead.

## 1. Start the API locally

Create a virtual environment, install the editable package, and run the FastAPI app with authentication disabled so the backend provisions a demo account automatically.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional: include OCR support with pip install -e ".[dev,ocr]"
export COMPAIR_REQUIRE_AUTHENTICATION=false
uvicorn compair_core.server.app:create_app --factory --reload
```

Keep the server running; it listens on `http://127.0.0.1:8000`.

## 2. Grab the demo session token

With authentication disabled, `GET /load_session` creates or refreshes the auto-provisioned user and returns a long-lived token. Save it for subsequent requests.

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

response = requests.get(f"{BASE_URL}/load_session", timeout=10)
session = response.json()
AUTH_HEADERS = {"auth-token": session["id"]}
print(f'User with user_id {session["user_id"]} is ready!')
```

## 3. Publish a reference document

Create a document that will serve as the reference material for later feedback. Set `is_published=true` so it can be surfaced during comparison.

```python
reference_text = "1. Define the goal.\n2. Gather assets.\n3. Announce launch."

reference = requests.post(
    f"{BASE_URL}/create_doc",
    headers=AUTH_HEADERS,
    data={
        "document_title": "Launch Checklist",
        "document_type": "txt",
        "document_content": "",
        "is_published": True,
    },
    timeout=10,
).json()

# Process once so the published doc has embeddings available for retrieval
requests.post(
    f"{BASE_URL}/process_doc",
    headers=AUTH_HEADERS,
    data={
        "doc_id": reference["document_id"],
        "doc_text": reference_text,
        "generate_feedback": False,
    },
    timeout=10,
)

print("Reference doc id:", reference["document_id"])
```

## 4. Draft a new document and request feedback

Create another document, then immediately call `POST /process_doc` with `generate_feedback=true`. Compair will embed the new content, look for relevant published references (including the one you just created), and produce AI feedback.

```python
candidate_text = (
    "Hi team, the launch happens tomorrow. "
    "Let's meet at noon to finalize assets."
)

candidate = requests.post(
    f"{BASE_URL}/create_doc",
    headers=AUTH_HEADERS,
    data={
        "document_title": "Launch Email Draft",
        "document_type": "txt",
        "document_content": "",
        "is_published": False,
    },
    timeout=10,
).json()

process = requests.post(
    f"{BASE_URL}/process_doc",
    headers=AUTH_HEADERS,
    data={
        "doc_id": candidate["document_id"],
        "doc_text": candidate_text,
        "generate_feedback": True,
    },
    timeout=10,
).json()
print("Processing result:", process)
```

If you're running the core edition locally, `process_doc` returns immediately (`{"task_id": null}`) after generating feedback.

## 5. Review the generated feedback

Fetch the feedback items associated with the candidate document. You should see model suggestions that reference the published checklist from step 3.

```python
feedback = requests.get(
    f"{BASE_URL}/documents/{candidate['document_id']}/feedback",
    headers=AUTH_HEADERS,
    timeout=10,
).json()

for item in feedback["feedback"]:
    print("-", item["feedback"])
```

That's it—you've created a published reference, drafted new content, and generated targeted feedback using Compair's local-only workflow.
