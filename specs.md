# Specs — Chatbot Festival Iguassu Inova 2026

Este arquivo contém as especificações completas para recriar (vibecode) este projeto em outra máquina Linux, usando outra instância do OpenCode com o modelo Nemotron 3.5 Super VL Free.

## 1. Ações necessárias do usuário (faça primeiro)

Antes de pedir à IA para gerar o código, o usuário deve:

1. **Clonar o repositório** (ou criar a pasta do projeto):
   ```bash
   git clone https://github.com/cesarbrod/iguassuinova.git
   cd iguassuinova
   ```
   Se não houver repositório, crie a pasta `iguassuinova/` e inicialize o git.

2. **Instalar dependências do sistema**:
   ```bash
   sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv curl
   ```

3. **Instalar dependências Python** (usando venv para isolamento):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install groq numpy scikit-learn requests beautifulsoup4 html2text flask
   ```

4. **Instalar o Ollama** (opcional, apenas para LLM local):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull gemma3:4b
   # ou: ollama pull phi3:mini
   ```
   > ⚠️ Evite o modelo `qwen3.5:4b`: possui modo "thinking" integrado que não pode ser desabilitado via API do Ollama, deixando a resposta vazia em RAG.

5. **Definir chave de API do Groq** (opcional, apenas se quiser usar nuvem):
   ```bash
   export GROQ_API_KEY=sua_chave_aqui
   ```

---

## 2. Visão geral do projeto

Chatbot educacional de RAG (Retrieval-Augmented Generation) que responde perguntas sobre o Festival Iguassu Inova 2026 com base em conteúdo 100% offline capturado do site oficial (https://festivaliguassuinova.com.br/). Não consulta a internet em tempo de execução.

Duas interfaces:
- **CLI** (`chatbot/chatbot.py`): terminal interativo em português do Brasil, suporte a Ollama (local) ou Groq (nuvem).
- **Web** (`web/app.py`): servidor Flask com interface responsiva (look and feel BrodTec), suporte a Ollama com seleção de modelo pela interface.

---

## 3. Estrutura de pastas e arquivos

```
iguassuinova/
├── README.md                      # Documentação principal
├── tutorial.md                    # Tutorial detalhado (instalação, programas, TF-IDF)
├── specs.md                       # Este arquivo: especificações para vibecode
├── LICENSE                        # GPLv3+
├── .gitignore                     # __pycache__, *.pyc, index/, .env
├── offline-content/               # Conteúdo offline capturado (Markdown)
│   ├── index.md
│   ├── faq.md
│   ├── palestrantes.md
│   ├── palestrante-perfil-*.md    # Perfis individuais de palestrantes
│   ├── mundos.md
│   ├── ficiencias.md
│   ├── itaipu-parquetec.md
│   ├── sapiens.md
│   ├── summit-tour.md
│   ├── fims.md
│   ├── rme.md
│   ├── o-festival-e-seu.md
│   ├── termos-de-uso.md
│   ├── politica-de-cookies.md
│   └── termos-de-participacao.md
├── chatbot/
│   ├── chatbot.py                 # Executável principal da versão CLI
│   ├── ingest.py                  # Cria índice de busca TF-IDF
│   └── scrape.py                  # Captura páginas do site oficial e reindexa
│   └── index/                     # Índice gerado (vectors.npy, chunks.json, etc.)
└── web/                           # Versão web (Flask + HTML/CSS/JS)
    ├── app.py                     # Servidor Flask + lógica RAG + Ollama
    ├── README.md                  # Documentação da versão web
    ├── static/
    │   ├── app.js                 # Frontend do chat
    │   └── style.css              # Estilos BrodTec
    └── templates/
        └── index.html             # Página principal com chat integrado
```

---

## 4. Programas e suas responsabilidades

### 4.1 `chatbot/ingest.py` — Construção do índice TF-IDF

**Entrada:** todos os arquivos `.md` em `offline-content/`.
**Saída:** arquivos em `chatbot/index/` (`vectors.npy`, `chunks.json`, `metadados.json`, `vectorizer.json`).

Funções principais:
- `read_md_files(content_dir)`: lê todos os markdowns, extrai o título (primeiro `# ` ou `## ` que não seja "Nossos Mundos"), retorna lista de dicionários com `source`, `title`, `text`, `path`.
- `chunk_text(text, size, overlap)`: divide o texto em chunks preservando seções de cabeçalho markdown (#, ##, ###). Remove imagens e blocos de cards de divulgação/post Instagram ruidosos. Retorna lista de strings com tamanho > 10.
- `build_index(docs)`: cria `TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1,2), max_df=0.9, min_df=1)`, ajusta nos chunks, salva `vectors.npy`, `chunks.json`, `metadados.json`, `vectorizer.json`.
- `load_index()`: carrega vetores, chunks e metadados dos arquivos salvos.

**Por que chunking por cabeçalhos:** mantém cada seção de palestrante (nome, data, horário, local, tema) coesa no mesmo chunk, evitando mistura de informações entre palestrantes diferentes.

### 4.2 `chatbot/scrape.py` — Captura automática do conteúdo offline

**Entrada:** site oficial https://festivaliguassuinova.com.br/.
**Saída:** arquivos Markdown em `offline-content/` e reindexação via `ingest.py`.

Opções de linha de comando:
- `--all`: páginas principais + perfis individuais de palestrantes.
- `--perfis`: apenas perfis individuais.
- `--limite N`: limita captura de perfis a N.
- `--so-reindexar`: apenas reexecuta `ingest.py` sem baixar nada.

Fluxo:
1. Baixa as 13 páginas principais (index, faq, palestrantes, o-festival-e-seu, ficiencias, itaipu-parquetec, sapiens, summit-tour, fims, rme, termos-de-uso, politica-de-cookies, termos-de-participacao).
2. Extrai links de perfis individuais da página de palestrantes (URLs contendo `lel_yazo_profile=`).
3. Converte HTML para Markdown com `html2text` + `BeautifulSoup` (remove scripts, styles, iframes; preserva links e imagens).
4. Aguarda 0,5s entre requisições (respeito ao servidor).
5. Reindexa com `ingest.py`.

### 4.3 `chatbot/chatbot.py` — Versão CLI

**Entrada:** índice TF-IDF + escolha de backend (Ollama ou Groq) + perguntas do usuário.
**Saída:** respostas impressas no terminal com formatação ANSI.

Fluxo principal:
1. Garante que o índice existe (cria via `ingest.py` se necessário).
2. Carrega vetores, chunks, metadados e vectorizer.
3. Pergunta ao usuário: "1. Modelo local via Ollama" ou "2. Groq".
4. Se Ollama: inicia `ollama serve` se não estiver rodando, lista modelos, permite seleção.
5. Se Groq: requer `GROQ_API_KEY` no ambiente.
6. Loop de perguntas:
   - Recupera chunks via `retrieve()` (top_k=15, threshold=0.02).
   - Se a pergunta menciona "palestra(s)" + mundo específico (ex: "palestras do mundo Itaipu Parquetec"), busca diretamente nos chunks de palestrantes daquele mundo via `buscar_palestras_por_mundo()` e coloca no topo do contexto.
   - Monta contexto com `format_context()`.
   - Envia ao LLM com SYSTEM_PROMPT + RAG_PROMPT.
   - Limpa a resposta com `clean_response()` (remove rótulos, loops, blocos Thinking, parágrafos duplicados, correção de datas "2dezena"→"2026", etc.).
   - Exibe com `wrap_text()` e `print_paginated()`.

Funções centralizadas (usadas também pela web):
- `extrair_mundo_da_pergunta(question)`: retorna o nome do mundo mencionado (ex: "Mundo Itaipu Parquetec") baseado em mapeamento `MUNDOS_NOMES`.
- `is_fonte_palestrante(fonte)`: True se `source == "palestrantes.md"` ou começa com `"palestrante-perfil"`.
- `buscar_palestras_por_mundo(mundo, chunks, metadados, limit=200)`: busca chunks de palestrantes que mencionam o mundo e têm data/horário/local/bloco/espaço/sala; monta atividade completa juntando chunks adjacentes do mesmo perfil.
- `format_answer(resultados, listagem_palestras=False, html=False)`: formata resposta. Se `listagem_palestras=True`, usa `_formatar_lista_palestras()` com negrito ANSI (CLI) ou `<bold>` (web).
- `ollama_chat(model, mensagens)`: consulta `http://localhost:11434/api/chat` com temperatura 0.3, num_predict 2000.
- `groq_chat(client, mensagens)`: consulta Groq com modelo `qwen/qwen3.8-27b`, temperatura 0.3, max_tokens 800.
- `clean_response(text)`: remove rótulos, loops de tokens, blocos Thinking, corrigi "2dezena"/"2de", remove instruções vazadas, deduplica parágrafos.
- `retrieve(question, vectors, chunks, metadados, vectorizer, top_k)`: busca por similaridade de cosseno TF-IDF.
- `format_context(results)`: monta string de contexto para o LLM.

### 4.4 `web/app.py` — Versão web (Flask)

**Entrada:** requisições HTTP + índice TF-IDF + modelo Ollama (opcional).
**Saída:** páginas HTML e respostas JSON da API.

Fluxo principal:
1. Carrega o índice na inicialização (`load_rag()`).
2. Auto-seleciona o primeiro modelo Ollama disponível se o servidor estiver rodando.
3. Rota `POST /api/chat`: recebe `{message}`, chama `rag_answer()`, retorna `{answer}`.
4. Rota `GET /api/ollama/status`: retorna `{running, models, selected, host, default_model}`.
5. Rota `POST /api/ollama/select`: seleciona modelo (ou `"model": ""` para desativar).
6. Rota `POST /api/ollama/start`: tenta iniciar `ollama serve`.

Linha de comando:
```bash
python3 app.py --modelo gemma3:4b --host 0.0.0.0 --porta 5000
```
Use `--modelo nenhum` para desativar o LLM local.

`rag_answer(question)`:
- Recupera chunks via `retrieve()` (top_k=5).
- Aplica regras de prioridade por palavra-chave (palestrante, mundos, acessibilidade, ingresso, datas/local).
- Se pergunta menciona palestra(s) + mundo, busca palestras daquele mundo e coloca no topo.
- Se `listagem_palestras=True`, pula o LLM e retorna diretamente o formato estruturado (evita que o LLM gere fragmentos aleatórios com contexto grande).
- Caso contrário, usa Ollama com SYSTEM_PROMPT + RAG_PROMPT, com fallback para chunks brutos se falhar.

### 4.5 `web/static/app.js` — Frontend da versão web

**Responsabilidades:**
- Envio de perguntas via `fetch` para `/api/chat`.
- Exibição de mensagens de usuário e bot com avatares.
- Indicador de digitação (typing).
- Perguntas sugeridas clicáveis.
- Carregamento do status do Ollama ao iniciar (`GET /api/ollama.status`).
- Seletor de modelo na barra lateral (`POST /api/ollama.select`).
- Conversão de `<bold>Nome</bold>` (enviado pelo backend) em `<strong>Nome</strong>` via `formatarNegrito()`.
- Escape de HTML para evitar XSS.
- Paginação automática com scroll para baixo.

### 4.6 `web/templates/index.html` — Página principal

Layout BrodTec com:
- Hero com dados do festival (datas, local, ingresso).
- Cards de informações rápidas.
- Sidebar com sobre o evento, perguntas sugeridas e card do modelo de linguagem.
- Área do chat com mensagens, input e botão enviar.
- Footer com licença e links.

---

## 5. Prompts do sistema

### SYSTEM_PROMPT (usado em chatbot.py e web/app.py)

```
Você é um assistente especializado no Festival Iguassu Inova 2026.
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
```

### RAG_PROMPT

```
Contexto (conteúdo offline do festival):
{context}

Pergunta do usuário: {question}

Responda diretamente, apenas com base no contexto fornecido.
```

---

## 6. Mapeamento de mundos (MUNDOS_NOMES)

Usado para extrair o mundo mencionado na pergunta:

| Chave na pergunta | Nome do mundo |
|-------------------|---------------|
| ficiencias / ficienicias | Mundo FICiências |
| itaipu parquetec / itaipu | Mundo Itaipu Parquetec |
| sapiens | Mundo Sapiens |
| summit tour / summit | Mundo Summit Tour |
| fims | Mundo FIMS |
| rme | Mundo RME |

---

## 7. Regras de prioridade na busca (rag_answer)

Quando a pergunta contém certas palavras, os resultados são reordenados:

| Palavra-chave na pergunta | Fontes priorizadas |
|---------------------------|--------------------|
| nome próprio (ex: Cesar Brod) | palestrantes.md primeiro |
| mundo / mundos | mundos.md e index.md primeiro |
| acessibilidade / acessível / libras / acompanhamento | faq.md primeiro |
| gratuito / ingresso / inscrição / inscricao | faq.md e termos-de-participacao.md primeiro |
| quando / onde / data / local | index.md e faq.md primeiro |

---

## 8. Formatação de resposta (listagem de palestras por mundo)

Quando a pergunta menciona "palestra(s)" + um mundo, o sistema lista todas as atividades daquele mundo no formato:

```
**Nome do Palestrante** | DD/MM/AAAA | HH:MM – HH:MM | Local · Mundo X | — Tema da atividade
```

- **Web:** negrito via `<bold>Nome</bold>` → convertido para `<strong>` no frontend.
- **CLI:** negrito via códigos ANSI `\033[1mNome\033[0m`.
- Cada evento separado por linha em branco (`\n\n`).

---

## 9. Como executar (resumo para a outra máquina)

```bash
# 1. Preparar ambiente
python3 -m venv .venv
source .venv/bin/activate
pip install groq numpy scikit-learn requests beautifulsoup4 html2text flask

# 2. (Opcional) Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b

# 3. Baixar/capturar conteúdo (já incluído no repo, mas pode reindexar)
python3 chatbot/scrape.py --so-reindexar

# 4. Executar CLI
cd chatbot
python3 chatbot.py
# Escolha 1 (Ollama) ou 2 (Groq)

# 5. Executar Web
cd web
python3 app.py --modelo gemma3:4b
# Acesse http://localhost:5000
```

---

## 10. Observações importantes

- O conteúdo do festival é 100% offline. O chatbot **não** consulta a internet em tempo de execução.
- O índice TF-IDF é gerado automaticamente na primeira execução ou com `chatbot/scrape.py --so-reindexar`.
- Evite o modelo `qwen3.5:4b` no Ollama (modo thinking integrado deixa resposta vazia).
- Modelos recomendados: `gemma3:4b`, `phi3:mini`, `llama3.1:8b`.
- Projeto educacional sob licença GPLv3+.
- Para informações oficiais: https://festivaliguassuinova.com.br/

---

## 11. Critérios de aceite (o que deve funcionar após o vibecode)

1. `python3 chatbot/ingest.py` cria o índice em `chatbot/index/` com vetores, chunks e metadados.
2. `python3 chatbot/chatbot.py` roda no terminal, permite escolher Ollama/Groq e responde perguntas sobre o festival.
3. `python3 web/app.py --modelo gemma3:4b` sobe o servidor Flask em `http://localhost:5000`.
4. Pergunta "Quais são as palestras do mundo Itaipu Parquetec?" retorna múltiplas palestras (incluindo Cesar Brod e Felipe Gonzalez) com nome, data, horário, local e tema.
5. Nomes de palestrantes aparecem em negrito (ANSI no CLI, `<strong>` na web).
6. Cada evento aparece em linha separada.
7. O card "Modelo de linguagem (local)" na web mostra se o Ollama está online e qual modelo está ativo.
