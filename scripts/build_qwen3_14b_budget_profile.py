"""Deterministically derive the qualified qwen3:14b budget-profile asset.

The inputs are exact, hash-verified upstream artifacts.  This script reads only
GGUF metadata; model tensor bytes are neither required nor accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import BinaryIO, ClassVar

MODEL_DIGEST = "sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
MODEL_LAYER_DIGEST = (
    "sha256:a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
)
GGUF_METADATA_PREFIX_BYTES = 33_554_432
GGUF_METADATA_PREFIX_SHA256 = (
    "7d9159485121a0e222f50eea45b05439cdc98822df1966a2b4c0023c7625d57a"
)
TEMPLATE_SHA256 = "ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
UNICODE_DATA_SHA256 = "95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d"
LLAMA_CPP_COMMIT = "7e4c0a96880dae4fc4268ad441f8a6446bd5460a"
OLLAMA_COMMIT = "d67ad83426633195089509347ffd4fe795120198"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class GGUFReader:
    _FORMATS: ClassVar[dict[int, str]] = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def _exact(self, size: int) -> bytes:
        raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError("truncated GGUF metadata")
        return raw

    def _scalar(self, value_type: int) -> object:
        if value_type == 8:
            size = struct.unpack("<Q", self._exact(8))[0]
            return self._exact(size).decode("utf-8", errors="strict")
        if value_type == 9:
            element_type = struct.unpack("<I", self._exact(4))[0]
            length = struct.unpack("<Q", self._exact(8))[0]
            return [self._scalar(element_type) for _ in range(length)]
        value_format = self._FORMATS.get(value_type)
        if value_format is None:
            raise ValueError(f"unsupported GGUF value type {value_type}")
        return struct.unpack(value_format, self._exact(struct.calcsize(value_format)))[
            0
        ]

    def metadata(self) -> dict[str, object]:
        if self._exact(4) != b"GGUF":
            raise ValueError("not a GGUF artifact")
        version = struct.unpack("<I", self._exact(4))[0]
        if version != 3:
            raise ValueError("unsupported GGUF version")
        _tensor_count = struct.unpack("<Q", self._exact(8))[0]
        metadata_count = struct.unpack("<Q", self._exact(8))[0]
        result: dict[str, object] = {}
        for _ in range(metadata_count):
            key = self._scalar(8)
            assert isinstance(key, str)
            value_type = struct.unpack("<I", self._exact(4))[0]
            if key in result:
                raise ValueError("duplicate GGUF metadata key")
            result[key] = self._scalar(value_type)
        return result


def _initializer_body(source: str, declaration: str) -> str:
    match = re.search(
        rf"{re.escape(declaration)}\s*=\s*\{{(.*?)\n\}};", source, re.DOTALL
    )
    if match is None:
        raise ValueError(f"missing {declaration}")
    return match.group(1)


def _pairs(source: str, declaration: str) -> list[list[int]]:
    body = _initializer_body(source, declaration)
    return [
        [int(left, 0), int(right, 0)]
        for left, right in re.findall(
            r"\{(0x[0-9A-Fa-f]+|\d+),\s*(0x[0-9A-Fa-f]+|\d+)\}", body
        )
    ]


def _values(source: str, declaration: str) -> list[int]:
    body = _initializer_body(source, declaration)
    return [int(value, 0) for value in re.findall(r"0x[0-9A-Fa-f]+|\d+", body)]


def build_profile(
    gguf_path: Path, template_path: Path, unicode_path: Path
) -> dict[str, object]:
    template_raw = template_path.read_bytes()
    unicode_raw = unicode_path.read_bytes()
    if _sha256(template_raw) != TEMPLATE_SHA256:
        raise ValueError("unexpected Ollama template bytes")
    if _sha256(unicode_raw) != UNICODE_DATA_SHA256:
        raise ValueError("unexpected llama.cpp Unicode data bytes")
    with gguf_path.open("rb") as stream:
        prefix = stream.read(GGUF_METADATA_PREFIX_BYTES)
        if (
            len(prefix) != GGUF_METADATA_PREFIX_BYTES
            or _sha256(prefix) != GGUF_METADATA_PREFIX_SHA256
        ):
            raise ValueError("unexpected qualified GGUF metadata-prefix bytes")
        stream.seek(0)
        metadata = GGUFReader(stream).metadata()

    expected = {
        "general.architecture": "qwen3",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.bos_token_id": 151643,
        "tokenizer.ggml.eos_token_id": 151645,
        "tokenizer.ggml.padding_token_id": 151643,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"unexpected qualified metadata for {key}")
    tokens = metadata.get("tokenizer.ggml.tokens")
    token_types = metadata.get("tokenizer.ggml.token_type")
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(tokens, list) or len(tokens) != 151_936:
        raise ValueError("unexpected tokenizer vocabulary")
    if not isinstance(token_types, list) or len(token_types) != len(tokens):
        raise ValueError("unexpected tokenizer token types")
    if not isinstance(merges, list) or len(merges) != 151_387:
        raise ValueError("unexpected tokenizer merges")
    if not all(isinstance(value, str) for value in (*tokens, *merges)):
        raise ValueError("non-string tokenizer data")
    if not all(isinstance(value, int) for value in token_types):
        raise ValueError("non-integer tokenizer token type")

    unicode_source = unicode_raw.decode("utf-8", errors="strict")
    profile: dict[str, object] = {
        "schema_version": "baseline-generation-budget-profile.v1",
        "qualification": {
            "model": "qwen3:14b",
            "model_digest": MODEL_DIGEST,
            "ollama_version": "0.32.14",
            "context_tokens": 32_768,
            "output_tokens": 1_024,
            "schema_consumes_prompt_tokens": False,
            "ollama_truncate": False,
        },
        "framing": {
            "system_prefix": "<|im_start|>system\n\n",
            "system_suffix": "<|im_end|>\n",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": (
                " /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
            "attestation_system": "COMPAIR_TEMPLATE_SYSTEM",
            "attestation_user": "COMPAIR_TEMPLATE_USER",
        },
        "tokenizer": {
            "model": "gpt2",
            "pre": "qwen2",
            "add_bos": False,
            "tokens": tokens,
            "merges": merges,
            "special_tokens": [
                [token, token_id]
                for token_id, (token, token_type) in enumerate(zip(tokens, token_types))
                if token_type in {2, 3, 4}
            ],
            "unicode_ranges_flags": _pairs(
                unicode_source,
                "const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags",
            ),
            "unicode_map_lowercase": _pairs(
                unicode_source,
                "const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase",
            ),
            "unicode_set_whitespace": _values(
                unicode_source,
                "const std::unordered_set<uint32_t> unicode_set_whitespace",
            ),
        },
        "origins": {
            "ollama_manifest": {
                "url": "https://registry.ollama.ai/v2/library/qwen3/manifests/14b",
                "sha256": MODEL_DIGEST.removeprefix("sha256:"),
            },
            "gguf_model_layer": {
                "url": "https://registry.ollama.ai/v2/library/qwen3/blobs/"
                + MODEL_LAYER_DIGEST,
                "digest": MODEL_LAYER_DIGEST,
                "metadata_prefix_bytes": GGUF_METADATA_PREFIX_BYTES,
                "metadata_prefix_sha256": GGUF_METADATA_PREFIX_SHA256,
                "metadata_components": {
                    "tokens_sha256": _sha256(_canonical(tokens)),
                    "token_types_sha256": _sha256(_canonical(token_types)),
                    "merges_sha256": _sha256(_canonical(merges)),
                },
                "license": "Apache-2.0",
            },
            "ollama_template_layer": {
                "url": "https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:"
                + TEMPLATE_SHA256,
                "sha256": TEMPLATE_SHA256,
                "license": "Apache-2.0",
                "template": template_raw.decode("utf-8", errors="strict"),
            },
            "ollama_source": {
                "commit": OLLAMA_COMMIT,
                "archive_sha256": "9ba34fce5fd63f331cdb52d45f427f2f72ec4dd3616424eff036e422be3deb8e",
                "license": "MIT",
                "source_hashes": {
                    "llm/llama_server.go": "264cc8bc64ce52162f689eb6052b6e6f2c9a92b5d30d693fee59c9e9c8c81429",
                    "server/prompt.go": "79411c4e15ff27fb8bac4dcd96076407e035ec407c0ab1a1840035f9668b4654",
                    "server/routes.go": "fc6a8a980168973dd1aec86aafde7ab8bbc12f56aa2f71e04a7773dc95916c42",
                },
            },
            "llama_cpp_source": {
                "commit": LLAMA_CPP_COMMIT,
                "archive_sha256": "8759ab3d3a92d86ba3ba24fab7e6adde08eaf2f941e6c79118373e4f41e0af8c",
                "license": "MIT",
                "source_hashes": {
                    "src/llama-vocab.cpp": "dab35ad158ccae5cb6064d960243ae7a6a045a09e0c4335f6491328750f8ad01",
                    "src/unicode.cpp": "aa75c6258a7e0d8ddc05476cbe68ce9baae99b8cf9ffad8a8ee545d176cb97da",
                    "src/unicode-data.cpp": UNICODE_DATA_SHA256,
                },
            },
        },
    }
    profile["profile_fingerprint"] = _sha256(_canonical(profile))
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--unicode-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = build_profile(args.gguf, args.template, args.unicode_data)
    args.output.write_bytes(_canonical(profile) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
