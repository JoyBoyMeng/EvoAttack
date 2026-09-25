#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge GOOD and ATTACK Chroma memory banks into one MIXED Chroma memory bank.

The script copies documents, metadatas, and embeddings from two existing
chromadb.PersistentClient databases into a new database. It preserves the ASB
storage format, so the mixed DB can be read by the original ASB code with:

    chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("langchain")

Typical usage:

    python scripts/scripts/merge_memory_banks.py \
        --good_db scripts/scripts/memory_banks/good_system_admin_agent_chromadb \
        --attack_db scripts/scripts/memory_banks/attack_system_admin_agent_chromadb \
        --out_db scripts/scripts/memory_banks/mixed_system_admin_agent_chromadb \
        --collection_name langchain \
        --overwrite

Optional ratio control:

    python scripts/scripts/merge_memory_banks.py \
        --good_sample 200 \
        --attack_sample 50

If --good_sample / --attack_sample are omitted, all records are copied.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb


DEFAULT_GOOD_DB = "scripts/scripts/memory_banks/good_system_admin_agent_chromadb"
DEFAULT_ATTACK_DB = "scripts/scripts/memory_banks/attack_system_admin_agent_chromadb"
DEFAULT_OUT_DB = "scripts/scripts/memory_banks/mixed_system_admin_agent_chromadb"
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 42


def to_plain_embedding(embedding: Any) -> List[float]:
    """Convert Chroma/numpy embedding objects into plain Python lists."""
    if embedding is None:
        raise ValueError("Encountered None embedding. Rebuild source DBs with embeddings first.")
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def load_collection_records(
    db_path: str,
    collection_name: str,
    bank_label: str,
) -> List[Dict[str, Any]]:
    """Read all ids/documents/metadatas/embeddings from one Chroma collection."""
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        raise FileNotFoundError(f"{bank_label} DB does not exist: {db_path}")

    client = chromadb.PersistentClient(path=str(db_path_obj))
    collection = client.get_collection(name=collection_name)

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = data.get("ids")
    docs = data.get("documents")
    metas = data.get("metadatas")
    embs = data.get("embeddings")

    ids = [] if ids is None else ids
    docs = [] if docs is None else docs
    metas = [] if metas is None else metas
    embs = [] if embs is None else embs

    if not (len(ids) == len(docs) == len(metas) == len(embs)):
        raise ValueError(
            f"Length mismatch in {bank_label} DB: "
            f"ids={len(ids)}, docs={len(docs)}, metas={len(metas)}, embs={len(embs)}"
        )

    records: List[Dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        metadata = dict(metas[i] or {})
        metadata["source_bank"] = bank_label

        # Normalize memory_type if the source script already provides it.
        # good bank usually has memory_type=good; attack bank has memory_type=attack.
        if "memory_type" not in metadata:
            metadata["memory_type"] = bank_label

        records.append({
            "id": str(doc_id),
            "document": docs[i],
            "metadata": metadata,
            "embedding": to_plain_embedding(embs[i]),
        })

    return records


def sample_records(
    records: List[Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
    label: str,
) -> List[Dict[str, Any]]:
    if sample_size is None:
        return records

    if sample_size < 0:
        raise ValueError(f"--{label}_sample must be non-negative or omitted.")

    if sample_size > len(records):
        raise ValueError(
            f"Requested --{label}_sample={sample_size}, "
            f"but only {len(records)} {label} records exist."
        )

    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:sample_size]


def make_unique_ids(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[List[float]]]:
    """Prefix IDs by source_bank and de-duplicate if needed."""
    used = set()
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    for idx, rec in enumerate(records):
        source_bank = rec["metadata"].get("source_bank", "unknown")
        base_id = f"{source_bank}__{rec['id']}"
        new_id = base_id
        suffix = 1
        while new_id in used:
            suffix += 1
            new_id = f"{base_id}__dup{suffix}"
        used.add(new_id)

        ids.append(new_id)
        docs.append(rec["document"])
        metas.append(rec["metadata"])
        embs.append(rec["embedding"])

    return ids, docs, metas, embs


def add_in_batches(collection: Any, ids: List[str], docs: List[str], metas: List[Dict[str, Any]], embs: List[List[float]], batch_size: int) -> None:
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embs[start:end],
        )
        print(f"[Write] {min(end, len(ids))}/{len(ids)} records")


def count_by_field(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        value = str(rec.get("metadata", {}).get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--good_db", default=DEFAULT_GOOD_DB)
    parser.add_argument("--attack_db", default=DEFAULT_ATTACK_DB)
    parser.add_argument("--out_db", default=DEFAULT_OUT_DB)
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION_NAME)

    parser.add_argument("--good_sample", type=int, default=None)
    parser.add_argument("--attack_sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    out_db = Path(args.out_db)
    if out_db.exists():
        if args.overwrite:
            print(f"[Info] Removing existing output DB: {out_db}")
            shutil.rmtree(out_db)
        else:
            raise FileExistsError(
                f"Output DB already exists: {out_db}\n"
                f"Use --overwrite if you want to rebuild it."
            )

    print("=" * 100)
    print("[Merge GOOD + ATTACK Memory Banks]")
    print(f"Good DB:   {args.good_db}")
    print(f"Attack DB: {args.attack_db}")
    print(f"Out DB:    {args.out_db}")
    print(f"Collection: {args.collection_name}")
    print("=" * 100)

    good_records = load_collection_records(
        db_path=args.good_db,
        collection_name=args.collection_name,
        bank_label="good",
    )
    attack_records = load_collection_records(
        db_path=args.attack_db,
        collection_name=args.collection_name,
        bank_label="attack",
    )

    print(f"[Load] Good records: {len(good_records)}")
    print(f"[Load] Attack records: {len(attack_records)}")

    good_records = sample_records(good_records, args.good_sample, args.seed, "good")
    attack_records = sample_records(attack_records, args.attack_sample, args.seed + 1, "attack")

    print(f"[Use] Good records: {len(good_records)}")
    print(f"[Use] Attack records: {len(attack_records)}")

    mixed_records = good_records + attack_records
    rng = random.Random(args.seed)
    rng.shuffle(mixed_records)

    ids, docs, metas, embs = make_unique_ids(mixed_records)

    print(f"[Mixed] Total records: {len(ids)}")
    print(f"[Mixed] memory_type counts: {json.dumps(count_by_field(mixed_records, 'memory_type'), ensure_ascii=False)}")
    print(f"[Mixed] source_bank counts: {json.dumps(count_by_field(mixed_records, 'source_bank'), ensure_ascii=False)}")

    out_client = chromadb.PersistentClient(path=str(out_db))
    out_collection = out_client.get_or_create_collection(name=args.collection_name)

    print("=" * 100)
    print("[Writing mixed DB]")
    add_in_batches(
        collection=out_collection,
        ids=ids,
        docs=docs,
        metas=metas,
        embs=embs,
        batch_size=args.batch_size,
    )

    print("=" * 100)
    print("[Done]")
    print(f"Mixed memories written: {len(ids)}")
    print(f"Chroma DB saved to: {out_db}")

    if ids:
        print("\n[Example ID]")
        print(ids[0])
        print("\n[Example metadata]")
        print(json.dumps(metas[0], indent=2, ensure_ascii=False))
        print("\n[Example document]")
        print(docs[0][:2000])


if __name__ == "__main__":
    main()
