#!/usr/bin/env python3
"""Chatbot RAG sobre o Festival Iguassu Inova — suporte a modelo local (Ollama) e Groq."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import termios
import textwrap
import time
import tty
import urllib.request
from pathlib import Path

import numpy as np
from groq import Groq

from ingest import CONTENT_DIR, INDEX_DIR, load_index, read_md_files, build_index

TOP_K = 15
SIMILARITY_THRESHOLD = 0.02
OLLAMA_HOST = "http://localhost:11434"

BOLD = "\033[1m"
RESET = "\033[0m"

SYSTEM_PROMPT = """Você é um assistente especializado no Festival Iguassu Inova 2026.
Responda perguntas em português do Brasil, de forma clara, objetiva e amigável, usando APENAS
o conteúdo fornecido no contexto abaixo (arquivos offline do site oficial do festival).

Regras:
- Se a resposta estiver no contexto, responda com base nele e cite a fonte (arquivo) quando relevante.
- Se a informação não estiver no contexto, diga que não tem essa informação nos materiais do festival e evite inventar.
- Não mencione que você é uma IA ou que está usando documentos fornecidos; fale como um guia do festival.
- Seja direto e útil. Mantenha o tom acolhedor, típico de um evento de inovação e cultura.
- O festival ocorre de 25 a 28 de novembro de 2026, no Itaipu Parquetec, Foz do Iguaçu/PR. Entrada gratuita.
- NÃO inclua rótulos como "Pergunta do usuário:", "Resposta ao usuário:", "Usuário:" ou "Assistente:" no início ou fim da sua resposta. Responda diretamente o conteúdo.
- NÃO repita a mesma informação em múltiplos parágrafos. Seja direto e evite redundância. Cada parágrafo deve trazer uma informação nova e relevante.
- Se a pergunta perguntar sobre uma pessoa específica (ex: "Cesar Brod"), busque essa pessoa no contexto e informe o que estiver disponível (nome, trilha/tema, descrição, cargo). Não diga que não há informação se o nome aparecer no contexto.
- Inclua TODOS os detalhes relevantes que estiverem no contexto (nomes, descrições, datas, horários, locais, valores, regras). Não resuma demais — se o contexto contiver descrições detalhadas, inclua-as. Se a resposta não estiver no contexto, informe que não dispõe dessa informação.
- NÃO use thinking mode. NÃO mostre seu processo de raciocínio. Apenas responda diretamente em português do Brasil.
"""

RAG_PROMPT = """Contexto (conteúdo offline do festival):
{context}

Pergunta do usuário: {question}

Responda diretamente, apenas com base no contexto fornecido."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(question, vectors, chunks, metadados, vectorizer, top_k=TOP_K):
    q_vec = vectorizer.transform([question]).toarray()[0]
    scores = []
    for i, doc_vec in enumerate(vectors):
        sim = cosine_similarity(q_vec, doc_vec)
        scores.append((sim, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, idx in scores[:top_k]:
        if sim < SIMILARITY_THRESHOLD:
            continue
        results.append(
            {
                "score": sim,
                "chunk": chunks[idx],
                "source": metadados[idx]["source"],
                "title": metadados[idx]["title"],
            }
        )
    return results


def format_context(results):
    parts = []
    for r in results:
        parts.append(f"[Fonte: {r['title']} ({r['source']})]\n{r['chunk']}\n")
    return "\n---\n".join(parts)


def get_terminal_size():
    return shutil.get_terminal_size(fallback=(80, 24))


def wrap_text(text, width=None):
    if width is None:
        width = get_terminal_size().columns
    paragraphs = text.split("\n\n")
    return "\n\n".join(
        textwrap.fill(p, width=width, break_long_words=False, replace_whitespace=False)
        for p in paragraphs
    )


def clean_response(text):
    # Remove rótulos repetitivos em qualquer lugar do texto
    text = re.sub(
        r"(Pergunta do usuário:|Resposta ao usuário:|Usuário:|Assistente:)\s*",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Remove blocos de pergunta/resposta repetidos (padrão "Pergunta... Resposta...")
    text = re.sub(
        r"\n\s*Pergunta do usuário:.*?\n\s*Resposta ao usuário:.*?(?=\n|$)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove sequências de caracteres repetidos (loops de tokens do modelo)
    # Ex: "000100010001..." ou "aaaaaa..."
    text = re.sub(r"(.)\1{5,}", r"\1\1\1", text)
    text = re.sub(r"(\S{2,})\1{3,}", r"\1", text)
    # Corrige alucinações de data como "2dezena" -> "2026", "2de" -> "2026", etc.
    text = re.sub(r"\b(\d)dezena\b", r"\1000", text)
    text = re.sub(r"\b(\d)de\b", r"\1000", text)
    # Corrige erros ortográficos comuns de alucinação do modelo
    text = text.replace("dispõo", "disponho")
    # Remove blocos "Thinking Process:" vazados pelo modelo Qwen3.5
    text = re.sub(
        r"\*\*Thinking Process:\*\*.*?(?=\n\n\*\*|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"Thinking Process:.*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove linhas de instruções vazadas pelo modelo
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("instruções:"):
            continue
        if stripped.lower().startswith("contexto (conteúdo offline"):
            continue
        if stripped.lower().startswith("pergunta do usuário:"):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Remove linhas duplicadas consecutivas
    lines = text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        if line.strip() != prev:
            deduped.append(line)
            prev = line.strip()
    text = "\n".join(deduped)
    # Remove parágrafos duplicados ou muito similares (>= 85% de similaridade)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    unique_paragraphs = []
    for para in paragraphs:
        is_duplicate = False
        for existing in unique_paragraphs:
            if _paragraphs_similar(para, existing, threshold=0.85):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_paragraphs.append(para)
    text = "\n\n".join(unique_paragraphs)
    return text.strip()


def _paragraphs_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    from difflib import SequenceMatcher

    if len(a) < 20 or len(b) < 20:
        return a == b
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def wait_for_space():
    if not sys.stdin.isatty():
        print()
        return
    print(f"{BOLD}Tecle espaço para continuar...{RESET}", end=" ", flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()


def print_paginated(text):
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        print(text)
        return
    _, height = get_terminal_size()
    lines_per_page = max(height - 4, 10)
    all_lines = text.split("\n")
    total = len(all_lines)
    for start in range(0, total, lines_per_page):
        print("\n".join(all_lines[start : start + lines_per_page]))
        if start + lines_per_page < total:
            wait_for_space()


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_is_running() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_ollama() -> subprocess.Popen | None:
    if ollama_is_running():
        return None
    print("Iniciando servidor Ollama (ollama serve)...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if ollama_is_running():
            print("Ollama está rodando.")
            return proc
        time.sleep(1)
    print("Aviso: Ollama não respondeu a tempo. Tente iniciar manualmente com 'ollama serve'.")
    return proc


def ollama_list_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_pull_model(model: str) -> bool:
    print(f"Baixando modelo {model} via Ollama (pode demorar)...")
    try:
        proc = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"Modelo {model} baixado com sucesso.")
            return True
        print(f"Erro ao baixar {model}: {proc.stderr}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao baixar modelo: {exc}")
        return False


def choose_ollama_model() -> str | None:
    models = ollama_list_models()
    if models:
        print("\nModelos locais disponíveis no Ollama:")
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        print("  0. Baixar um novo modelo")
        escolha = input("Escolha o número do modelo (ou 0 para baixar): ").strip()
        if escolha == "0":
            novo = input("Nome do modelo para baixar (ex: llama3.1:8b, phi3:mini, gemma3:4b): ").strip()
            if not novo:
                return None
            if ollama_pull_model(novo):
                return novo
            return None
        try:
            idx = int(escolha) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                if "qwen3.5" in selected:
                    print(
                        "\n⚠️  Aviso: O modelo qwen3.5 usa modo 'thinking' integrado e pode não "
                        "funcionar bem com RAG via Ollama. Recomendamos usar phi3:mini ou "
                        "llama3.1:8b para melhores resultados.\n"
                    )
                return selected
        except ValueError:
            pass
        print("Escolha inválida.")
        return None
    else:
        print("Nenhum modelo local encontrado no Ollama.")
        recomendado = input("Baixar modelo recomendado (llama3.1:8b)? [s/N]: ").strip().lower()
        if recomendado in {"s", "sim"}:
            if ollama_pull_model("llama3.1:8b"):
                return "llama3.1:8b"
        else:
            manual = input("Nome do modelo para baixar: ").strip()
            if manual and ollama_pull_model(manual):
                return manual
    return None


def ollama_chat(model: str, mensagens: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": mensagens,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2000},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resultado = json.loads(resp.read().decode())
        message = resultado.get("message", {})
        content = message.get("content", "").strip()
        # Qwen3.5 coloca o raciocínio no campo 'thinking' e deixa 'content' vazio
        if not content and message.get("thinking"):
            content = message.get("thinking", "").strip()
        if not content:
            return ""
        # Remove blocos "Thinking Process:" e similares
        content = re.sub(
            r"\*\*Thinking Process:\*\*.*?(?=\n\n\*\*|\Z)",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r"Thinking Process:.*",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = content.strip()
        return content
    except Exception as exc:  # noqa: BLE001
        return f"Desculpe, ocorreu um erro ao consultar o modelo local: {exc}"
        return f"Desculpe, ocorreu um erro ao consultar o modelo local: {exc}"


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def groq_chat(client: Groq, mensagens: list[dict]) -> str:
    try:
        resposta = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=mensagens,
            temperature=0.3,
            max_tokens=800,
        )
        return resposta.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        return f"Desculpe, ocorreu um erro ao consultar o modelo Groq: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Garante que o índice existe
    if not (INDEX_DIR / "vectors.npy").exists():
        print("Índice não encontrado. Criando índice dos conteúdos offline...")
        docs = read_md_files(CONTENT_DIR)
        build_index(docs)

    vectors, chunks, metadados = load_index()

    from sklearn.feature_extraction.text import TfidfVectorizer

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

    # Escolha do backend
    print("\n" + "=" * 60)
    print("🎪 Chatbot do Festival Iguassu Inova 2026")
    print("=" * 60)
    print("Escolha o backend de LLM:")
    print("  1. Modelo local via Ollama (totalmente offline, sem custo)")
    print("  2. Groq (nuvem, requer GROQ_API_KEY)")
    escolha = input("Digite 1 ou 2: ").strip()

    use_local = escolha == "1"

    ollama_proc = None
    groq_client = None
    local_model = None

    if use_local:
        ollama_proc = start_ollama()
        local_model = choose_ollama_model()
        if not local_model:
            print("Nenhum modelo local selecionado. Saindo.")
            if ollama_proc:
                ollama_proc.terminate()
            return
        print(f"\nUsando modelo local: {local_model}")
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERRO: Defina a variável de ambiente GROQ_API_KEY antes de executar com Groq.")
            print("Exemplo: export GROQ_API_KEY=sua_chave_aqui")
            return
        groq_client = Groq(api_key=api_key)
        print("\nUsando backend: Groq (modelo qwen/qwen3-32b)")

    print("\nPergunte qualquer coisa sobre o festival (datas, ingressos,")
    print("palestrantes, mundos, FAQ, acessibilidade, etc.).")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.\n")

    historico = []

    while True:
        try:
            pergunta = input(f"{BOLD}Você:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté a próxima! 👋")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit", "q"}:
            print("\nAté a próxima! 👋")
            break

        # Recupera chunks relevantes
        resultados = retrieve(pergunta, vectors, chunks, metadados, vectorizer, top_k=TOP_K)
        contexto = format_context(resultados)

        mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
        mensagens.append(
            {"role": "user", "content": RAG_PROMPT.format(context=contexto, question=pergunta)}
        )

        if use_local:
            texto = ollama_chat(local_model, mensagens)
        else:
            texto = groq_chat(groq_client, mensagens)

        if not texto:
            texto = (
                "Desculpe, o modelo selecionado não conseguiu gerar uma resposta "
                "para esta pergunta. Tente outro modelo (ex: phi3:mini) ou o backend Groq."
            )

        texto_limpo = clean_response(texto)
        texto_wrapped = wrap_text(texto_limpo)
        print(f"\n{BOLD}Bot:{RESET}")
        print_paginated(texto_wrapped)

        historico.append({"role": "user", "content": pergunta})
        historico.append({"role": "assistant", "content": texto})

    if ollama_proc:
        ollama_proc.terminate()


if __name__ == "__main__":
    main()
