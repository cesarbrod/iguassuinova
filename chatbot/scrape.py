#!/usr/bin/env python3
"""Scraper automático do site do Festival Iguassu Inova 2026.

Captura as páginas oficiais do site (https://festivaliguassuinova.com.br/),
converte para Markdown e salva em offline-content/. Ao final, reindexa o
conteúdo TF-IDF executando ingest.py.

Uso:
    python3 chatbot/scrape.py                    # páginas principais
    python3 chatbot/scrape.py --perfis           # + perfis individuais de palestrantes
    python3 chatbot/scrape.py --perfis --limite 50  # limita a N perfis individuais
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import html2text

BASE_URL = "https://festivaliguassuinova.com.br/"
CONTENT_DIR = Path(__file__).resolve().parent.parent / "offline-content"

PAGINAS_PRINCIPAIS = [
    ("index", ""),
    ("faq", "faq.html"),
    ("palestrantes", "palestrantes.html"),
    ("o-festival-e-seu", "o-festival-e-seu.html"),
    ("ficiencias", "ficiencias.html"),
    ("itaipu-parquetec", "itaipu-parquetec.html"),
    ("sapiens", "sapiens.html"),
    ("summit-tour", "summit-tour.html"),
    ("fims", "fims.html"),
    ("rme", "rme.html"),
    ("termos-de-uso", "termos-de-uso.html"),
    ("politica-de-cookies", "politica-de-cookies.html"),
    ("termos-de-participacao", "termos-de-participacao.html"),
]

CONVERTER = html2text.HTML2Text()
CONVERTER.ignore_links = False
CONVERTER.ignore_images = False
CONVERTER.body_width = 0  # não quebra linhas longas


def limpar_nome_arquivo(nome: str) -> str:
    nome = nome.strip().lower()
    nome = re.sub(r"[^a-z0-9\-]+", "-", nome)
    nome = re.sub(r"-{2,}", "-", nome).strip("-")
    return nome or "pagina"


def html_para_markdown(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav de header/footer repetitivos
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    texto = CONVERTER.handle(str(soup))

    # Remove linhas em branco excessivas
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Adiciona cabeçalho com fonte
    cabecalho = f"# Fonte: {url}\n\n"
    return cabecalho + texto.strip() + "\n"


def capturar_pagina(caminho: str, nome_arquivo: str) -> bool:
    url = urljoin(BASE_URL, caminho)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  ERRO ao baixar {url}: {exc}")
        return False

    md = html_para_markdown(resp.text, url)
    destino = CONTENT_DIR / f"{nome_arquivo}.md"
    destino.write_text(md, encoding="utf-8")
    print(f"  OK: {destino.name} ({len(md)} chars)")
    return True


def extrair_links_perfis() -> list[str]:
    """Extrai URLs de perfis individuais da página de palestrantes."""
    url = urljoin(BASE_URL, "palestrantes.html")
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  ERRO ao baixar {url}: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "lel_yazo_profile=" in href:
            links.append(urljoin(BASE_URL, href))
    # Remove duplicados preservando ordem
    vistos = set()
    unicos = []
    for l in links:
        if l not in vistos:
            vistos.add(l)
            unicos.append(l)
    return unicos


def capturar_perfis(limite: int | None = None) -> int:
    links = extrair_links_perfis()
    print(f"  Encontrados {len(links)} perfis individuais de palestrantes.")
    if limite is not None:
        links = links[:limite]
        print(f"  Limitando a {limite} perfis.")

    capturados = 0
    for i, url in enumerate(links, 1):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERRO perfil {i}/{len(links)} {url}: {exc}")
            continue

        # Tenta extrair o número do perfil para nomear o arquivo
        m = re.search(r"lel_yazo_profile=(\d+)", url)
        sufixo = m.group(1) if m else limpar_nome_arquivo(urlparse(url).path or "perfil")
        nome = f"palestrante-perfil-{sufixo}"

        md = html_para_markdown(resp.text, url)
        destino = CONTENT_DIR / f"{nome}.md"
        destino.write_text(md, encoding="utf-8")
        print(f"  OK: {destino.name} ({i}/{len(links)})")
        capturados += 1
        time.sleep(0.5)  # respeitoso com o servidor

    return capturados


def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper do site do Festival Iguassu Inova 2026")
    parser.add_argument(
        "--perfis",
        action="store_true",
        help="Também captura perfis individuais de palestrantes (palestrantes/?lel_yazo_profile=NNN).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Faz scrape de tudo: páginas principais + perfis individuais de palestrantes (equivalente a rodar com --perfis).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Número máximo de perfis individuais a capturar (ex: 50).",
    )
    parser.add_argument(
        "--so-reindexar",
        action="store_true",
        help="Apenas reexecuta o ingest.py (não baixa nada).",
    )
    args = parser.parse_args()

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    if args.so_reindexar:
        print("Reindexando conteúdos existentes...")
    else:
        print("=== Capturando páginas principais ===")
        ok = 0
        for nome, caminho in PAGINAS_PRINCIPAIS:
            if capturar_pagina(caminho, nome):
                ok += 1
            time.sleep(0.5)
        print(f"Páginas principais capturadas: {ok}/{len(PAGINAS_PRINCIPAIS)}")

        if args.perfis or args.all:
            print("\n=== Capturando perfis individuais de palestrantes ===")
            n = capturar_perfis(limite=args.limite)
            print(f"Perfis capturados: {n}")

        print("\n=== Reindexando ===")

    # Reindexa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import build_index, read_md_files

    docs = read_md_files(CONTENT_DIR)
    build_index(docs)
    print("\nConcluído. Conteúdo atualizado e índice recriado.")


if __name__ == "__main__":
    main()
