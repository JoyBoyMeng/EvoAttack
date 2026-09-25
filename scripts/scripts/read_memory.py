import argparse
import json
import os
import chromadb


def parse_where(value):
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--where must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--where must decode to a JSON object")
    return parsed


def memory_text(document, metadata, text_key):
    if document:
        return document
    if isinstance(metadata, dict):
        return metadata.get(text_key) or metadata.get("data") or metadata.get("text") or metadata.get("text_lemmatized")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        required=True,
        help="Chroma memory directory, e.g. combined_attack_gpt-4o-mini"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to read. Defaults to all collections."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of memories to print per collection."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset within each collection."
    )
    parser.add_argument(
        "--where",
        type=parse_where,
        default=None,
        help='Optional Chroma metadata filter JSON, e.g. \'{"namespace":"adaptive_target_normal"}\'.'
    )
    parser.add_argument(
        "--text-key",
        default="data",
        help="Metadata key to use as text when Chroma document is empty. Default: data."
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print full metadata for each memory."
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print collection counts."
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)

    print("=" * 80)
    print(f"Reading Chroma DB: {db_path}")
    print("=" * 80)

    client = chromadb.PersistentClient(path=db_path)

    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    if args.collection:
        collection_names = [args.collection]
    else:
        collection_names = [col.name if hasattr(col, "name") else col for col in collections]

    for collection_name in collection_names:
        print("\n" + "=" * 80)
        print(f"Collection: {collection_name}")
        print("=" * 80)

        collection = client.get_collection(collection_name)
        count = collection.count()

        print(f"Total memories: {count}")

        if count == 0 or args.count_only:
            continue

        get_kwargs = {"include": ["documents", "metadatas"]}
        if args.limit is not None:
            get_kwargs["limit"] = args.limit
        if args.offset:
            get_kwargs["offset"] = args.offset
        if args.where:
            get_kwargs["where"] = args.where

        data = collection.get(**get_kwargs)

        ids = data["ids"]
        documents = data["documents"]
        metadatas = data["metadatas"]

        for i in range(len(ids)):
            print("\n" + "-" * 80)
            print(f"Memory #{i}")
            print(f"ID: {ids[i]}")

            if args.show_metadata:
                print("\nMetadata:")
                print(json.dumps(metadatas[i], indent=2, ensure_ascii=False))

            print("\nText:")
            print(memory_text(documents[i], metadatas[i], args.text_key))


if __name__ == "__main__":
    main()
