# Qualified qwen3:14b budget-profile provenance

## Supported tuple

This profile supports one tuple and no implied successors:

- model name: `qwen3:14b`;
- manifest digest: `sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`;
- Ollama runtime: exactly `0.32.14`;
- configured context: 32,768 tokens;
- reserved output: 1,024 tokens;
- adapter contract: `baseline-generation-ollama-http.v2`.

Every other model name, manifest digest, runtime version, context size, or
output reservation is unsupported until it receives a separate qualified
profile. In particular, a newer Ollama runtime cannot reuse this profile.

## Packaged result

| Asset | SHA-256 |
| --- | --- |
| `qwen3-14b-ollama-0.32.14.profile.json` | `af00a090678da236d35203b01cdb929543e30bdcbc59749efe60e8ad20e1a284` |
| profile fingerprint | `69ccd81f6ba8e62a34961559390c170879315431ba58a96cf99ba90ac035bda9` |
| packaged `qwen3-14b.LICENSE.md` | `b87250e4478f8f0673bc61cd397e402ac33ca3a6f23a7ad9512618ab33f18fbe` |
| packaged `THIRD_PARTY_NOTICES.md` | `ec023b446ab00d24ec486be900b9145583d61bb0356fd05c9f1d2ef27ab72408` |
| upstream license layer, before the packaged trailing LF | `d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12` |

The profile fingerprint is SHA-256 over compact, sorted-key UTF-8 JSON of the
entire profile object before adding its `profile_fingerprint` member. Runtime
loading verifies both the packaged byte hash and that independently calculated
fingerprint.

The profile contains the minimum runtime inputs used by the counter: the exact
token strings, merge order, special-token IDs, pinned Unicode category and
lowercase tables, exact native template bytes, fixed two-message framing,
schema-treatment decision, and source provenance. It contains no model tensor
data.

## Exact upstream records

### qwen3:14b registry artifacts

The compact manifest response from
`https://registry.ollama.ai/v2/library/qwen3/manifests/14b` is 859 bytes with
SHA-256 `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
It names these immutable layers:

| Media | Digest | Bytes |
| --- | --- | ---: |
| config | `sha256:78b3b822087d5199783c8203553a5a92ce5eb7b683a5a81003f8efea9b399d74` | 488 |
| GGUF model | `sha256:a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e` | 9,276,184,896 |
| native Go template | `sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4` | 1,723 |
| license | `sha256:d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12` | 11,338 |
| parameters | `sha256:cff3f395ef3756ab63e58b0ad1b32bb6f802905cae1472e6a12034e4246fbbdb` | 120 |

Only the first 33,554,432 bytes of the GGUF layer are needed to reproduce the
profile. That range has SHA-256
`7d9159485121a0e222f50eea45b05439cdc98822df1966a2b4c0023c7625d57a`;
the complete GGUF metadata ends at byte 5,932,816. The metadata identifies
GGUF v3, architecture `qwen3`, pre-tokenizer `qwen2`, 151,936 token strings,
151,936 token types, and 151,387 merges. Component hashes over compact JSON
are embedded in the profile itself.

### Ollama 0.32.14

Tag `v0.32.14` resolves to commit
`d67ad83426633195089509347ffd4fe795120198`. The source archive SHA-256 is
`9ba34fce5fd63f331cdb52d45f427f2f72ec4dd3616424eff036e422be3deb8e`;
its MIT license SHA-256 is
`5934ed2ce0d15154bcdb9c85203210abac0da4314af34081e36df4599f90b226`.

Relevant source byte hashes are:

- `server/prompt.go`: `79411c4e15ff27fb8bac4dcd96076407e035ec407c0ab1a1840035f9668b4654`;
- `server/routes.go`: `fc6a8a980168973dd1aec86aafde7ab8bbc12f56aa2f71e04a7773dc95916c42`;
- `llm/llama_server.go`: `264cc8bc64ce52162f689eb6052b6e6f2c9a92b5d30d693fee59c9e9c8c81429`.

Those sources establish that the native Go template renders the messages,
`format` is supplied separately as the structured-output schema/grammar, and
the llama-server `/tokenize` endpoint performs prompt counting with special
token parsing. The production request explicitly sets `truncate:false`.
Attestation uses Ollama's non-inference `_debug_render_only` mode with the
actual schema present and compares the complete rendered result to the
profile. This detects a runtime selecting a different template path, including
native Jinja selection.

### Ollama-pinned llama.cpp

Ollama pins llama.cpp build `b10434`, commit
`7e4c0a96880dae4fc4268ad441f8a6446bd5460a`. The source archive SHA-256 is
`8759ab3d3a92d86ba3ba24fab7e6adde08eaf2f941e6c79118373e4f41e0af8c`;
its MIT license SHA-256 is
`94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`.

Relevant source byte hashes are:

- `src/llama-vocab.cpp`: `dab35ad158ccae5cb6064d960243ae7a6a045a09e0c4335f6491328750f8ad01`;
- `src/unicode.cpp`: `aa75c6258a7e0d8ddc05476cbe68ce9baae99b8cf9ffad8a8ee545d176cb97da`;
- `src/unicode-data.cpp`: `95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d`.

The local counter ports the pinned Qwen2 Unicode splitting, GPT-2 byte
encoding, special-token partition, BPE rank queue, and fallback behavior. It
was compared with a tokenizer-only GGUF and the exact pinned
`llama-tokenize`: all golden cases and 100 seeded randomized Unicode,
punctuation, control-character, multiline, and special-token cases matched
token-for-token. No inference was run.

## Deterministic derivation

1. Download and hash the exact manifest.
2. Download bytes `0-33554431` of the exact GGUF layer, the exact template
   layer, the exact license layer, and `src/unicode-data.cpp` from the pinned
   llama.cpp commit. The builder verifies the GGUF metadata-prefix, template,
   and Unicode-source hashes; verify the license hash above before continuing.
3. Run:

   ```text
   python scripts/build_qwen3_14b_budget_profile.py \
     --gguf qwen3-14b.gguf.header \
     --template qwen3-14b.template \
     --unicode-data unicode-data.cpp \
     --output qwen3-14b-ollama-0.32.14.profile.json
   ```

4. Verify the output byte hash and profile fingerprint in the packaged-result
   table. A mismatch is a stop condition, not a reason to normalize or repair
   an asset.

## License implications

The qwen3 tokenizer metadata and template originate from an Apache-2.0 model
distribution. The complete upstream Apache-2.0 text and Alibaba Cloud notice
are packaged, and the third-party notice identifies that the profile is a
modified, tensor-free derivative. The Ollama and llama.cpp behavioral ports
retain their MIT copyright and permission notices in
`THIRD_PARTY_NOTICES.md`. The repository's own MIT license remains applicable
to original Compair code; it does not replace these third-party terms.
