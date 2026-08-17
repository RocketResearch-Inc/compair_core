# Baseline control-plane deployment trust

This note is operational guidance for the frozen
`baseline-control-plane.v1` contract. It does not change the protocol bytes.

## Transport trust model

Core accepts a control-plane request only in one of these cases:

1. The ASGI connection scheme is HTTPS.
2. The immediate connected peer IP is inside an explicitly configured proxy
   CIDR and that proxy supplies one unambiguous original scheme of HTTPS.
3. `COMPAIR_BASELINE_CONTROL_PLANE_ALLOW_INSECURE_LOOPBACK=true`, the ASGI
   connection is HTTP, the immediate peer IP is loopback, the peer is not a
   configured proxy, and no proxy headers are present. This exception is for
   direct local development only.

Configure trusted TLS terminators with a comma-separated IP/CIDR list, for
example:

```text
COMPAIR_BASELINE_CONTROL_PLANE_TRUSTED_PROXY_ALLOWLIST=10.40.2.15/32,2001:db8:40::15/128
```

Hostnames, DNS resolution, wildcards, `Host`, `Forwarded` client/host values,
`X-Forwarded-For`, `X-Real-IP`, and similar advertised client addresses never
establish trust. An untrusted peer cannot make HTTP safe by supplying proxy
headers. A trusted proxy must supply exactly one consistent `proto=https`
attestation in `Forwarded`, `X-Forwarded-Proto`, or both. Conflicting, chained,
or malformed scheme values fail closed. An invalid proxy allowlist also fails
closed for proxied HTTP.

The proxy must:

- terminate authenticated TLS and connect from a stable allowlisted address;
- remove client-supplied forwarding headers, then write its own single-valued
  scheme header;
- never expose the direct local-development listener beyond loopback;
- enforce request limits no larger than Core's advertised limits; and
- use the same narrow proxy CIDRs in the ASGI server's forwarded-header
  configuration. Do not enable a wildcard such as Uvicorn
  `--forwarded-allow-ips='*'`, because middleware can rewrite ASGI connection
  metadata before Core evaluates it.

## Logging and request handling

Control-plane writes are POST-only and JSON bodies are strict UTF-8. Core
checks declared and streamed byte limits before parsing, rejects duplicate
keys at any object depth, rejects non-finite JSON numbers, and only then runs
schema validation and RFC 8785 canonicalization.

At the TLS terminator, ingress, ASGI server, APM agent, and WAF:

- disable request-body capture for `/baseline/control/v1/*` and
  `/baseline/control/admin/v1/*`;
- log only the route template and status, not a raw URL or query string;
- redact authorization/cookie headers and control-plane identifiers;
- disable exception/request dumps and sampling that can retain bodies; and
- keep access-log retention and deletion aligned with the staged-content
  retention policy.

Control-plane clients must never place raw diffs, file contents, credentials,
or sensitive identifiers in a URL. Core responses, durable job status, and
errors contain only bounded protocol metadata and sanitized codes.
