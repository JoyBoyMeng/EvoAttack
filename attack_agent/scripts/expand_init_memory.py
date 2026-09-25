"""Expand a completed clean init-memory Chroma collection by replication."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

import chromadb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate each record in a completed init-memory collection. "
            "Copies preserve text and attack-tool metadata without LLM calls."
        )
    )
    parser.add_argument("--db", required=True, help="Chroma init-memory directory")
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--copies",
        type=int,
        required=True,
        help="Total copies per original record, including the original.",
    )
    parser.add_argument(
        "--expected_source_count",
        type=int,
        required=True,
        help="Fail unless the collection currently contains exactly this many source records.",
    )
    return parser.parse_args()


def expand_init_memory(
    *,
    db_path: Path,
    collection_name: str,
    copies: int,
    expected_source_count: int,
) -> int:
    if copies < 1:
        raise ValueError("--copies must be at least 1.")
    if expected_source_count < 1:
        raise ValueError("--expected_source_count must be positive.")

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(collection_name)
    source_count = collection.count()
    if source_count != expected_source_count:
        raise ValueError(
            f"Expected exactly {expected_source_count} source records, found {source_count}. "
            "Refusing to replicate an already-expanded or incomplete init memory."
        )

    data = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = list(data.get("ids") if data.get("ids") is not None else [])
    documents = list(data.get("documents") if data.get("documents") is not None else [])
    metadatas = list(data.get("metadatas") if data.get("metadatas") is not None else [])
    embeddings = list(data.get("embeddings") if data.get("embeddings") is not None else [])
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings) == source_count):
        raise ValueError("Source collection returned inconsistent ids/documents/metadata/embeddings.")

    # Keep one canonical init seed for deterministic D1-D5 replay while all
    # replicated records remain eligible for normal target-memory retrieval.
    metadatas = [dict(metadata or {}) for metadata in metadatas]
    for metadata in metadatas:
        metadata["init_copy_index"] = 1
    collection.update(ids=ids, metadatas=metadatas)

    source_sequences = [
        int((metadata or {}).get("memory_sequence_ns", 0))
        for metadata in metadatas
    ]
    next_sequence = max(source_sequences, default=0)
    new_ids: list[str] = []
    new_documents: list[Any] = []
    new_metadatas: list[dict[str, Any]] = []
    new_embeddings: list[list[float]] = []

    for copy_index in range(2, copies + 1):
        for source_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            next_sequence += 1
            copied_metadata = dict(metadata)
            copied_metadata["init_copy_index"] = copy_index
            copied_metadata["memory_sequence_ns"] = next_sequence
            new_ids.append(f"{source_id}__init_copy_{copy_index}__{uuid.uuid4().hex[:8]}")
            new_documents.append(document)
            new_metadatas.append(copied_metadata)
            new_embeddings.append(list(embedding))

    if new_ids:
        add_kwargs: dict[str, Any] = {
            "ids": new_ids,
            "metadatas": new_metadatas,
            "embeddings": new_embeddings,
        }
        if any(document is not None for document in new_documents):
            add_kwargs["documents"] = new_documents
        collection.add(**add_kwargs)

    final_count = collection.count()
    expected_final_count = expected_source_count * copies
    if final_count != expected_final_count:
        raise RuntimeError(
            f"Expected {expected_final_count} records after replication, found {final_count}."
        )
    return final_count


def main() -> None:
    args = parse_args()
    final_count = expand_init_memory(
        db_path=Path(args.db),
        collection_name=args.collection,
        copies=args.copies,
        expected_source_count=args.expected_source_count,
    )
    print(f"Expanded init memory to {final_count} records.")


if __name__ == "__main__":
    main()
