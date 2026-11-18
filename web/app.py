#!/usr/bin/env python3
"""Servidor web local para o Chatbot do Festival Iguassu Inova 2026.

Backend: Flask. Frontend: HTML/CSS/JS com look and feel BrodTec (brodtec.com).
Integra a lógica RAG existente (ingest.py + chatbot.py).
Respostas baseadas no contexto offline via busca TF-IDF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Garante que o diretório pai (raiz do projeto) esteja no início do path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify

from chatbot.ingest import CONTENT_DIR, INDEX_DIR, load_index, read_md_files, build_index
from chatbot.chatbot import (
    SYSTEM_PROMPT,
    RAG_PROMPT,
    retrieve,
    format_context,
    clean_response,
    cosine_similarity,
)

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


# ---------------------------------------------------------------------------
# RAG answer (baseado apenas no contexto offline)
# ---------------------------------------------------------------------------

def rag_answer(question: str) -> str:
    """Gera uma resposta RAG para a pergunta usando apenas o contexto offline."""
    if _vectorizer is None:
        load_rag()

    print(f"[DEBUG] Iniciando retrieve para: {question[:50]}", flush=True)
    resultados = retrieve(question, _vectors, _chunks, _metadados, _vectorizer, top_k=5)
    print(f"[DEBUG] Retrieve retornou {len(resultados)} resultados", flush=True)

    if not resultados:
        return "Não encontrei informações sobre isso nos conteúdos offline do festival."

    # Se a pergunta menciona um nome próprio (palestrante), prioriza palestrantes.md
    import re as _re
    nomes_encontrados = _re.findall(r"\b[A-ZÁÉÍÓÚÇ][a-záéíóúç]+ [A-ZÁÉÍÓÚÇ][a-záéíóúç]+\b", question)
    if nomes_encontrados:
        resultados_palestrantes = [r for r in resultados if r["source"] == "palestrantes.md"]
        resultados_outros = [r for r in resultados if r["source"] != "palestrantes.md"]
        resultados = resultados_palestrantes + resultados_outros

    # Se a pergunta é sobre mundos, prioriza mundos.md e index.md
    if any(palavra in question.lower() for palavra in ["mundo", "mundos"]):
        resultados_mundos = [r for r in resultados if r["source"] in ("mundos.md", "index.md")]
        resultados_outros = [r for r in resultados if r["source"] not in ("mundos.md", "index.md")]
        resultados = resultados_mundos + resultados_outros

        # Se não há resultado de mundos.md/index.md, busca direta por palavra-chave
        if not resultados_mundos:
            for arquivo in ["mundos.md", "index.md"]:
                caminho = CONTENT_DIR / arquivo
                if caminho.exists():
                    texto_arquivo = caminho.read_text(encoding="utf-8")
                    if "Mundo" in texto_arquivo:
                        linhas = texto_arquivo.split("\n")
                        linhas_mundos = [l for l in linhas if ("Mundo FICiências" in l or "Mundo Itaipu" in l or "Mundo Sapiens" in l or "Mundo Summit" in l or "Mundo FIMS" in l or "Mundo RME" in l)]
                        if linhas_mundos:
                            chunk = "\n".join(linhas_mundos[:8])
                            resultados.insert(0, {
                                "score": 1.0,
                                "chunk": chunk,
                                "source": arquivo,
                                "title": arquivo.replace(".md", "").title(),
                            })
                            break

    # Se a pergunta é sobre acessibilidade, prioriza faq.md
    if any(palavra in question.lower() for palavra in ["acessibilidade", "acessível", "libras", "acompanhamento"]):
        resultados_faq = [r for r in resultados if r["source"] == "faq.md"]
        resultados_outros = [r for r in resultados if r["source"] != "faq.md"]
        resultados = resultados_faq + resultados_outros

    # Se a pergunta é sobre ingresso/gratuito, prioriza faq.md e termos-de-participacao.md
    if any(palavra in question.lower() for palavra in ["gratuito", "ingresso", "inscrição", "inscricao"]):
        resultados_faq = [r for r in resultados if r["source"] in ("faq.md", "termos-de-participacao.md")]
        resultados_outros = [r for r in resultados if r["source"] not in ("faq.md", "termos-de-participacao.md")]
        resultados = resultados_faq + resultados_outros

    # Se a pergunta é sobre datas/local, prioriza index.md e faq.md
    if any(palavra in question.lower() for palavra in ["quando", "onde", "data", "local"]):
        resultados_index = [r for r in resultados if r["source"] in ("index.md", "faq.md")]
        resultados_outros = [r for r in resultados if r["source"] not in ("index.md", "faq.md")]
        resultados = resultados_index + resultados_outros

    # Monta resposta baseada no contexto (sem depender de modelo externo)
    return _format_answer(resultados)


def _format_answer(resultados: list[dict]) -> str:
    """Formata uma resposta concisa e bem estruturada a partir dos chunks."""
    import re

    partes = []
    fontes_vistas = set()

    # Se há resultado de palestrantes.md, mostra apenas esse (já é suficiente)
    palestrantes_result = [r for r in resultados if r["source"] == "palestrantes.md"]
    outros_result = [r for r in resultados if r["source"] != "palestrantes.md"]

    # Se há resultado de mundos.md, mostra apenas esse (já é suficiente)
    mundos_result = [r for r in outros_result if r["source"] == "mundos.md"]
    outros_result = [r for r in outros_result if r["source"] != "mundos.md"]

    # Ordem de prioridade: palestrantes > mundos > faq > outros
    resultados_ordenados = palestrantes_result + mundos_result + outros_result

    for r in resultados_ordenados:
        fonte = f"{r['title']} ({r['source']})"
        if fonte in fontes_vistas:
            continue
        fontes_vistas.add(fonte)

        # Limpa o chunk: remove marcações de markdown excessivas e quebras de linha
        chunk = r["chunk"]
        # Remove imagens markdown
        chunk = re.sub(r"!\[.*?\]\(.*?\)", "", chunk)
        # Remove links markdown, mantendo o texto
        chunk = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", chunk)
        # Remove múltiplas quebras de linha
        chunk = re.sub(r"\n{3,}", "\n", chunk)
        # Remove múltiplos espaços
        chunk = re.sub(r"[ \t]+", " ", chunk)
        chunk = chunk.strip()

        if len(chunk) < 20:
            continue

        # Para perguntas sobre uma pessoa específica, prioriza o chunk que menciona o nome
        if r["source"] == "palestrantes.md" and len(chunk) > 500:
            linhas = chunk.split("\n")
            linhas_filtradas = []
            captura = False
            for linha in linhas:
                if linha.startswith("### ") or linha.startswith("## "):
                    if captura:
                        break
                    captura = True
                if captura:
                    linhas_filtradas.append(linha)
            if linhas_filtradas:
                chunk = "\n".join(linhas_filtradas[:15])

        partes.append(f"**{r['title']}**: {chunk[:400]}")

        # Se já temos uma fonte principal relevante, não precisamos de mais
        if r["source"] in ("palestrantes.md", "mundos.md", "faq.md"):
            break

    if not partes:
        return "Não encontrei informações relevantes sobre isso nos conteúdos offline do festival."

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