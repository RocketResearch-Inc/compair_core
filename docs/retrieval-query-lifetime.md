# Retrieval query protected-lifetime policy

Phase 2A passes `retrieval_query` from the `/process_doc` request to the
document-processing task and then keeps it in memory through the retrieval
engine invocation. In deployments where `process_document_task.delay(...)` is
a Celery call, the raw query is serialized into the broker message. Core does
not put it in the task return value, task-status metadata, result backend,
application logs, retrieval traces, or the durable corpus/index tables.

The broker message is therefore part of the query's protected lifetime. Core
now fails before dispatch unless every configured broker endpoint uses
authenticated, certificate-verified TLS. For Redis this means a `rediss://`
URL containing a password (or Redis credential provider) and
`ssl_cert_reqs=required`; for RabbitMQ it means `amqps://`, credentials (or a
configured client certificate/key), and `broker_use_ssl.cert_reqs=required`.
Celery task protocol v2 is required and `result_extended` must be false.

Explicit-query dispatch uses fixed redacted `argsrepr` and `kwargsrepr` values.
Celery's task-sent/task-received events and native worker "task received" log
therefore see redacted representations even when events are enabled. The raw
query remains in the encrypted task body because the worker must consume it.

`COMPAIR_RETRIEVAL_QUERY_ALLOW_INSECURE_LOCAL_TRANSPORT=true` is an explicit
development-only exception. It is accepted only for direct/eager execution,
Celery's in-memory test transport, a local Redis Unix socket, or a broker whose
host is syntactically loopback (`localhost`, `.localhost`, or a loopback IP).
It never permits an insecure remote or filesystem-backed broker. Protocol v2
and non-extended results remain mandatory for Celery even under this exception.

Deployments must also restrict broker access to API and worker principals,
avoid starting workers with custom signal/event handlers that inspect raw task
arguments, and expire or acknowledge messages promptly. The worker holds the
text only until the retrieval call finishes. Retries may extend that lifetime
by retaining the same protected broker payload.

All persistent and observable query provenance uses only:

- SHA-256 of the exact UTF-8 query bytes;
- query length in Unicode characters;
- origin: `explicit`, `legacy_derived`, or `absent`.

Changing to an encrypted ephemeral payload store can shorten broker exposure in
a later phase, but must be an explicit transport change with expiry, deletion,
and retry tests. Phase 2B1 does not silently persist the query under another
name.

The `/capabilities` and `/health` responses expose a credential-free
`retrieval_query_transport` object. Its status is `safe`, `unavailable`, or
`locally_overridden`; it never includes the broker URL, username, password, or
query.
