"""Build a LangChain-backed FAISS knowledge base for PlantGuard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

SUPPORTED_EXTS = {".txt", ".md", ".json"}
DEFAULT_INDEX_NAME = "knowledge_index"


def iter_documents(root: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path, path.read_text(encoding="utf-8", errors="ignore")


def load_json_entries(content: str) -> List[tuple[str, str]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    entries: List[tuple[str, str]] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                question = item.get("question") or item.get("prompt") or ""
                answer = item.get("answer") or item.get("response") or ""
                text = (question + "\n" + answer).strip()
                if text:
                    entries.append((f"item_{idx}", text))
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                entries.append((str(key), value.strip()))
    return entries


def build_corpus(input_dir: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    documents: List[Document] = []
    for path, raw_text in iter_documents(input_dir):
        relative = path.relative_to(input_dir)
        if path.suffix.lower() == ".json":
            entries = load_json_entries(raw_text)
            for key, snippet in entries:
                for idx, chunk in enumerate(splitter.split_text(snippet)):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": str(relative),
                                "chunk_id": f"{key}::{idx}",
                            },
                        )
                    )
            continue

        for idx, chunk in enumerate(splitter.split_text(raw_text)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(relative),
                        "chunk_id": f"chunk_{idx}",
                    },
                )
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LangChain FAISS index for PlantGuard knowledge base")
    parser.add_argument("--input", default="data/knowledge_base", help="Directory containing knowledge documents")
    parser.add_argument("--output", default="models/text/knowledge_base", help="Directory to store generated index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name for HuggingFaceEmbeddings")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Character chunk size for documents")
    parser.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="Name used when saving the FAISS index")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    documents = build_corpus(input_dir, splitter)
    if not documents:
        raise RuntimeError("No text chunks were produced. Check input directory and supported formats.")

    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    vectorstore = FAISS.from_documents(documents, embeddings)

    output_dir = Path(args.output)
    if output_dir.exists():
        for file in output_dir.glob("*"):
            if file.is_file():
                file.unlink()
            else:
                import shutil

                shutil.rmtree(file)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(output_dir), index_name=args.index_name)

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedding_model": args.model,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "index_name": args.index_name,
                "num_documents": len(documents),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved LangChain FAISS index to {output_dir}")
    print(f"Total documents: {len(documents)}")


if __name__ == "__main__":
    main()
