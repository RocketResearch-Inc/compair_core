"""Manual smoke check for a local baseline BGE/FastEmbed HTTP service."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from compair_core.compair.retrieval.embedding import (
    assess_baseline_embedding,
    require_configured_baseline_embedding_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dimension", type=int, default=384)
    args = parser.parse_args()

    settings = SimpleNamespace(
        baseline_embedding_provider="http",
        baseline_embedding_endpoint=args.endpoint,
        baseline_embedding_model=args.model,
        baseline_embedding_revision=args.revision,
        baseline_embedding_dimension=args.dimension,
        baseline_embedding_timeout_seconds=10.0,
        baseline_embedding_batch_size=2,
        baseline_embedding_allow_insecure_loopback=True,
    )
    capability = assess_baseline_embedding(settings)
    if not capability.available:
        print(json.dumps(capability.as_dict(), sort_keys=True))
        return 1

    adapter = require_configured_baseline_embedding_adapter(settings)
    vectors = adapter.embed(("public smoke probe one", "public smoke probe two"))
    output = capability.as_dict()
    output.update(
        {
            "vector_count": len(vectors),
            "vector_dtype": str(vectors[0].dtype) if vectors else None,
            "vector_dimension": len(vectors[0]) if vectors else None,
        }
    )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
