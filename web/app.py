#!/usr/bin/env python3
"""Servidor web local para o Chatbot do Festival Iguassu Inova 2026.

Backend: Flask. Frontend: HTML/CSS/JS com look and feel BrodTec (brodtec.com).
Integra a lógica RAG existente (ingest.py + chatbot.py).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Garante que o diretório pai (raiz do projeto) esteja no início do path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify, send_from_directory

from chatbot.ingest import CONTENT_DIR, INDEX_DIR, load_index, read_md_files, build_index
from chatbot.chatbot import (
    SYSTEM_PROMPT,
    RAG_PROMPT,
    retrieve,
    format_context,
    clean_response,
    wrap_text,
    cosine_similarity,
)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:  # pragma: no cover
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Estado global do RAG
# ---------------------------------------------------------------------------
_vectorizer: TfidfVectorizer | None = None
_vectors = None
_chunks: list[str] = []
_metadados: list[dict] = []


def ensure_index() -> None:
    """Garante que o índice exista; se não, cria."""
    if not (INDEX_DIR / "vectors.npy").exists():
        print("Índice não encontrado. Criando índice dos conteúdos offline...")
        docs = read_md_files(CONTENT_DIR)
        build_index(docs)


def load_rag() -> None:
    """Carrega o índice TF-IDF na memória."""
    global _vectorizer, _vectors, _chunks, _metadados
    ensure_index()
    vectors, chunks, metadados = load_index()

    with open(INDEX_DIR / "vectorizer.json", encoding="utf-8") as f:
        vcfg = json.load(f)
    vectorizer = TfidfVectorizer(
        lowercase=vcfg["lowercase"],
        stop_words=None,
        ngram_range=tuple(vcfg["ngram_range"]),
        max_df=vcfg["max_df"],
        min_df=vcfg["min_df"],
        vocabulary=vcfg["vocabulary"],
    )
    vectorizer.idf_ = np.array(vcfg["idf"])

    _vectorizer = vectorizer
    _vectors = vectors
    _chunks = chunks
    _metadados = metadados
    print(f"RAG carregado: {len(_chunks)} chunks de {len(set(m['source'] for m in _metadados))} arquivos.")


def rag_answer(question: str) -> str:
    """Gera uma resposta RAG para a pergunta usando o contexto offline."""
    if _vectorizer is None:
        load_rag()

    print(f"[DEBUG] Iniciando retrieve para: {question[:50]}", flush=True)
    resultados = retrieve(question, _vectors, _chunks, _metadados, _vectorizer, top_k=6)
    print(f"[DEBUG] Retrieve retornou {len(resultados)} resultados", flush=True)

    if not resultados:
        return "Não encontrei informações sobre isso nos conteúdos offline do festival."

    # Monta resposta baseada no contexto (sem depender de modelo externo)
    partes = []
    fontes_vistas = set()
    for r in resultados:
        fonte = f"{r['title']} ({r['source']})"
        if fonte in fontes_vistas:
            continue
        fontes_vistas.add(fonte)
        # Limpa o chunk para remover quebras de linha excessivas
        chunk_limpo = " ".join(r["chunk"].split())
        partes.append(f"**{r['title']}**: {chunk_limpo[:500]}")

    texto = "\n\n".join(partes)
    return clean_response(texto)


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/test", methods=["POST"])
def api_test():
    print("[DEBUG] api_test chamado", flush=True)
    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        print(f"[DEBUG] Erro JSON: {e}", flush=True)
        return jsonify({"error": str(e)}), 400
    print(f"[DEBUG] Dados: {data}", flush=True)
    return jsonify({"ok": True, "received": data})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    print("[DEBUG] api_chat chamado", flush=True)
    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        print(f"[DEBUG] Erro ao fazer parse do JSON: {e}", flush=True)
        return jsonify({"error": "JSON inválido"}), 400
    pergunta = (data.get("message") or "").strip()
    print(f"[DEBUG] Pergunta: {pergunta[:50]}", flush=True)
    if not pergunta:
        return jsonify({"error": "Pergunta vazia"}), 400

    try:
        resposta = rag_answer(pergunta)
        print(f"[DEBUG] Resposta gerada: {resposta[:80]}...", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Erro no rag_answer: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        resposta = f"Desculpe, ocorreu um erro ao processar sua pergunta: {exc}"

    return jsonify({"answer": resposta})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_rag()
    print("Servidor web iniciado em http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
