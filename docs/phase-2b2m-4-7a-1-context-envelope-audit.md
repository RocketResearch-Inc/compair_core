# Phase 2B2M.4.7A.1 production generation-input envelope audit

## Decision

The mandatory stop condition applies. The production path is not provably
fail-closed for every serialized prompt that can reach native Ollama.

Core rejects clearly oversized raw message content before `POST /api/chat`,
but the admission check does not include native chat-template and role framing.
It also does not establish whether, or how, Ollama's structured-output schema
contributes to the model context. Consequently, an input at the current
31,744-byte message-content boundary can be sent without a proof that the
fully rendered input plus the reserved 1,024 output tokens fits the configured
32,768-token context.

No production behavior, protocol, migration, fixture, semantic audit, or
existing examination asset was changed. The context-stress companion was not
created.

## Audited source identity

- Remote: `https://github.com/RocketResearch-Inc/compair_core.git`
- Branch: `feature/baseline-v1`
- Remote and checked-out commit:
  `ee75fbb0ac53eba6fa6b2fc290a88a829ad004c0`
- Starting worktree: clean
- Frozen 120-case examination SHA-256:
  `2f1d8d204de06173fbfbe7fabf00aeb5771ef9869c09cbd959b2e7b4789d5863`
- Frozen semantic-audit SHA-256:
  `6c6778ed3e007caaa4f7d76c2efa6b37863938dcb5f43170e8230369b2eb1167`

The two frozen hashes were verified before source files were read.

## Exact provider-visible data path

1. The v2 run endpoint reads at most 8,100,000 request bytes and
   `parse_run_submission` requires an explicit `raw_git_diff_v1`,
   UTF-8 retrieval query. It verifies the declared byte count and SHA-256 over
   the exact query bytes.
2. `BaselineRunJobService.submit` records query length, byte length, and hash,
   then places the exact query text in a canonical JSON plaintext protected by
   AES-256-GCM. Its authenticated associated data binds the job, source,
   publication, protocol, and query metadata.
3. `BaselineDocumentRunExecutor` decrypts the payload and rechecks exact
   character length, byte length, and SHA-256. It passes the unchanged query to
   document-level persistent `baseline_v1` retrieval.
4. Persistent retrieval ranks whole sibling-repository files, takes the frozen
   top-six cut, and selects one to four unique ordered evidence items within
   the shared 16,000-character content budget.
5. Evidence persistence stores the exact selected content and the frozen
   renderer output
   `Repository file: {repository}/{relative_path}\n\n{selected_content}`,
   with characters, hashes, rank data, and one-based ordinal.
6. `BaselineGenerationInputAdapter` reloads rows using
   `ORDER BY ordinal ASC`; reconstructs and verifies every renderer output,
   byte/character count, hash, contiguous ordinal, and corpus/index identity;
   and combines those unchanged strings with the authoritative whole
   `document.content` as `BaselineGenerationInput.source_text`.
7. The native provider creates one system message and one user message. The
   user message is the authoritative source followed by every stored renderer
   output under a one-based `Ordered evidence N:` heading, without clipping,
   reranking, or dropping evidence.
8. `generate` first reattests with `GET /api/version` and `GET /api/tags`.
   The message-size check runs afterward in `_chat`, so an oversized job can
   contact the configured runtime but does not reach inference.
9. The provider compact-serializes a nonstreaming `POST /api/chat` JSON body
   with the exact packaged schema as `format`, the two messages, and
   `temperature=0`, fixed seed, `num_ctx=32768`, and
   `num_predict=1024`.

The raw Git diff is the retrieval query and is integrity-preserved, but it is
not copied into the generation messages. Generation receives the authoritative
changed document and the ordered retrieved evidence.

## Applicable production bounds

### Control and retrieval

| Surface | Exact bound |
| --- | ---: |
| v2 run request body | 8,100,000 bytes |
| raw-query UTF-8 | 8,000,000 bytes |
| raw-query Python characters | 8,000,000 characters |
| sibling source file | 200,000 UTF-8 bytes |
| ranking projection per sibling file | 12,000 characters |
| retrieval cut | 6 candidates |
| selected evidence | 1–4 items |
| selected content per item | no independent cap beyond the shared budget; at most 16,000 characters |
| selected content per job | 16,000 characters |
| selected content per job, UTF-8 consequence | at most 64,000 bytes |

The evidence renderer adds 17 characters for `Repository file: `, one slash,
two newlines, an ASCII repository name of at most 128 characters, and a
relative path of at most 4,096 characters. Therefore:

- maximum renderer overhead per item: 4,244 characters;
- maximum renderer-output characters across four items:
  `16,000 + 4 × 4,244 = 32,976`;
- maximum renderer-output UTF-8 bytes across four items:
  `64,000 + 4 × (17 + 128 + 1 + 4 × 4,096 + 2) = 130,128`.

These are admitted retrieval/persistence bounds, not generation-safe bounds.

### Authoritative source text

`control_document` generation reads the complete authorized
`document.content` from an unbounded database `Text` column. Repository
registration checks authorization and identity but imposes no Core byte or
character limit on that document. The document creation path sanitizes NUL
characters but imposes no content-size contract.

Accordingly, the applicable Core bounds for
`BaselineGenerationInput.source_text` are:

- UTF-8 bytes: unbounded by the Core contract;
- Python characters: unbounded by the Core contract.

A valid admitted control job can therefore assemble source plus ordered
evidence far beyond the configured context.

### Qualified native Ollama envelope

The exact fixed values below use the qualified `qwen3:14b` model name and
seed zero. Dynamic source and renderer strings are excluded.

| Evidence items | System chars/bytes | Fixed user chars/bytes | Fixed message chars/bytes | Fixed request-body bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 786 | 61 | 847 | 2,106 |
| 2 | 786 | 83 | 869 | 2,131 |
| 3 | 786 | 105 | 891 | 2,156 |
| 4 | 786 | 127 | 913 | 2,181 |

The packaged output-schema file is 1,450 bytes and has SHA-256
`fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27`.
Its compact JSON representation inside the request body is 1,050 bytes. The
fixed request-body overhead other than raw system/user content is 1,259,
1,262, 1,265, or 1,268 bytes for one through four evidence items. JSON escaping
of dynamic message content can add further body bytes.

The provider request-body limit defaults to 256,000 bytes and may be configured
from 4,096 through 8,000,000 bytes. The qualified context is 32,768 tokens,
with 1,024 output tokens reserved, leaving 31,744 nominal input tokens.

Before building the request body, Core currently permits:

```text
len(system.encode("utf-8")) + len(user.encode("utf-8")) <= 31,744
```

It separately rejects a compact JSON request body larger than the configured
request-body limit. Neither check modifies the input.

## Incompatible bounds and truncation finding

The source-text contract is unbounded, selected renderer output can reach
32,976 characters or 130,128 UTF-8 bytes, and the control query can contain
8,000,000 bytes. Those admission bounds intentionally serve different stages
and are much larger than the qualified generation envelope.

For obviously large generation input, the raw-content check raises
`provider_request_too_large` before `POST /api/chat`; ordered evidence is
not clipped or dropped. However, byte-counting only the two message contents
does not calculate the tokens in Ollama's fully rendered native prompt. The
check excludes at least role/chat-template framing and has no versioned rule
for schema-to-context accounting. The request-body byte limit is not a token
limit. Core therefore cannot prove that every request it does send fits, and
cannot rule out provider-side truncation at the boundary.

This contradicts the required fail-closed guarantee, so no stress-fixture size
band can honestly be frozen from the current production envelope.

## Narrowest deterministic input-budget contract

Add one provider-boundary preflight without changing v2 admission, raw-diff
integrity, retrieval, ordering, persistence, or the frozen control-plane
contract:

1. Version and hash the exact qualified tokenizer, native chat template, role
   framing, and structured-schema context treatment for the attested
   model/runtime contract.
2. Assemble the exact unchanged source and ordered renderer messages.
3. Locally calculate the token count of the fully rendered provider prompt,
   including every fixed component that consumes context.
4. Before attestation or any evidence-bearing request, require
   `input_tokens + configured_output_tokens <= configured_context_tokens`.
5. Independently compact-serialize the exact HTTP body and require its UTF-8
   length not to exceed `maximum_request_bytes`.
6. On either failure, return one stable terminal budget error. Never truncate
   source, clip the raw diff, shorten renderer output, reorder or drop
   evidence, or rely on Ollama truncation.

The resulting versioned preflight budget—not arbitrary fixture padding—would
define compact, medium, and high safe-envelope bands for a later 12-case
companion.
