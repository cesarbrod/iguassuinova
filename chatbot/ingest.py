#!/usr/bin/env python3
"""Ingestão de conteúdos offline: carrega markdowns, faz chunks e cria índice TF-IDF."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CONTENT_DIR = Path(__file__).resolve().parent.parent / "offline-content"
INDEX_DIR = Path(__file__).resolve().parent / "index"
CHUNK_SIZE = 600   # caracteres aproximados por chunk
CHUNK_OVERLAP = 100


def read_md_files(content_dir: Path) -> list[dict]:
    docs = []
    for md_path in sorted(content_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        # Extrai o título do conteúdo (primeiro cabeçalho # ou ## após a linha Fonte:)
        title = md_path.stem.replace("-", " ").title()
        for linha in text.split("\n"):
            linha = linha.strip()
            if linha.startswith("# Fonte:"):
                continue
            if linha.startswith("# "):
                title = linha[2:].strip()
                break
            if linha.startswith("## ") and not linha.startswith("## Nossos"):
                title = linha[3:].strip()
                break
        docs.append({"source": md_path.name, "title": title, "text": text, "path": str(md_path)})
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Remove linhas de navegação repetidas e imagens para focar no texto útil
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Quebra por parágrafos primeiro
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # Overlap: pega o final do chunk anterior
            current = para
        if len(current) > size:
            # chunk muito longo, parte em pedaços
            while len(current) > size:
                chunks.append(current[:size])
                current = current[size - overlap:]
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c.strip()) > 30]


def build_index(docs: list[dict]):
    INDEX_DIR.mkdir(exist_ok=True)
    all_chunks = []
    metadados = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadados.append(
                {
                    "source": doc["source"],
                    "title": doc["title"],
                    "path": doc["path"],
                }
            )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2), max_df=0.9, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    np.save(INDEX_DIR / "vectors.npy", tfidf_matrix.toarray())
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(INDEX_DIR / "metadados.json", "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)
    with open(INDEX_DIR / "vectorizer.json", "w", encoding="utf-8") as f:
        # TfidfVectorizer não é serializável em JSON puro; guardamos vocab e idf
        json.dump(
            {
                "vocabulary": vectorizer.vocabulary_,
                "idf": vectorizer.idf_.tolist(),
                "max_df": vectorizer.max_df,
                "min_df": vectorizer.min_df,
                "ngram_range": list(vectorizer.ngram_range),
                "lowercase": vectorizer.lowercase,
            },
            f,
            ensure_ascii=False,
        )

    print(f"Índice criado: {len(all_chunks)} chunks de {len(docs)} arquivos.")
    return vectorizer, tfidf_matrix, all_chunks, metadados


def load_index():
    vectors = np.load(INDEX_DIR / "vectors.npy")
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(INDEX_DIR / "metadados.json", encoding="utf-8") as f:
        metadados = json.load(f)
    return vectors, chunks, metadados


if __name__ == "__main__":
    docs = read_md_files(CONTENT_DIR)
    build_index(docs)
