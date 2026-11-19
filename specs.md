# Specs — Chatbot Festival Iguassu Inova 2026

Este arquivo contém as especificações completas para recriar (vibecode) este projeto do zero em outra máquina Linux, usando outra instância do OpenCode com o modelo Nemotron 3.5 Super VL Free. Não é necessário clonar nenhum repositório existente: crie a estrutura de pastas e arquivos conforme descrito abaixo.

## 1. Ações necessárias do usuário (faça primeiro)

Antes de pedir à IA para gerar o código, o usuário deve preparar o ambiente:

1. **Criar a pasta do projeto** e inicializar o git (opcional):
   ```bash
   mkdir -p iguassuinova
   cd iguassuinova
   git init
   ```

2. **Instalar dependências do sistema**:
   ```bash
   sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv curl
   ```

3. **Criar ambiente virtual e instalar dependências Python**:
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

O festival ocorre de 25 a 28 de novembro de 2026, no Itaipu Parquetec, Foz do Iguaçu/PR. Entrada gratuita.

---

## 3. Estrutura de pastas e arquivos a criar

```
iguassuinova/
├── README.md                      # Documentação principal
├── tutorial.md                    # Guia detalhado de instalação e programas
├── specs.md                       # Este arquivo: especificações para vibecode
├── LICENSE                        # Licença GPLv3+
├── .gitignore                     # Ignora __pycache__, *.pyc, index/, .env
├── offline-content/               # Conteúdo offline capturado (Markdown)
│   ├── index.md                   # Página principal (datas, local, palestrantes, patrocinadores)
│   ├── faq.md                     # Perguntas frequentes
│   ├── palestrantes.md            # Palestrantes confirmados com biografias
│   ├── palestrante-perfil-*.md    # Perfis individuais de palestrantes (capturados do site)
│   ├── mundos.md                  # Seis mundos temáticos
│   ├── ficiencias.md              # Mundo FICiências
│   ├── itaipu-parquetec.md        # Mundo Itaipu Parquetec
│   ├── sapiens.md                 # Mundo Sapiens
│   ├── summit-tour.md             # Mundo Summit Tour
│   ├── fims.md                    # Mundo FIMS
│   ├── rme.md                     # Mundo RME
│   ├── o-festival-e-seu.md        # Chamada "O Festival é Seu"
│   ├── termos-de-uso.md           # Termos de uso
│   ├── politica-de-cookies.md     # Política de cookies
│   └── termos-de-participacao.md  # Termos de participação
├── chatbot/
│   ├── chatbot.py                 # Executável principal da versão CLI
│   ├── ingest.py                  # Cria índice de busca TF-IDF dos conteúdos
│   └── scrape.py                  # Captura páginas do site oficial e reindexa
│   └── index/                     # Índice gerado automaticamente (vectors.npy, chunks.json, metadados.json, vectorizer.json)
└── web/                           # Versão web (Flask + HTML/CSS/JS)
    ├── app.py                     # Servidor Flask e lógica RAG + LLM local (Ollama)
    ├── README.md                  # Documentação específica da versão web
    ├── static/
    │   ├── app.js                 # Frontend: chat, status do Ollama, seletor de modelo
    │   └── style.css              # Estilos BrodTec
    └── templates/
        └── index.html             # Página principal com chat integrado
```

---

## 4. Detalhamento dos programas

### 4.1 `chatbot/ingest.py` — Construção do índice TF-IDF

**Propósito:** Ler todos os arquivos Markdown de `offline-content/`, dividir em chunks e criar índice de busca TF-IDF.

**Funções que devem existir:**

- `read_md_files(content_dir)`:
  - Lê todos os arquivos `.md` do diretório.
  - Extrai o título de cada documento: primeiro cabeçalho `# ` ou `## ` que não seja "Nossos Mundos".
  - Retorna lista de dicionários com chaves: `source` (nome do arquivo), `title`, `text`, `path`.

- `chunk_text(text, size=600, overlap=100)`:
  - Remove imagens markdown (`![...](...)`).
  - Remove blocos de "Cards de Divulgação" e "Post para Instagram" (ruído visual).
  - Divide o texto em seções por cabeçalhos markdown (#, ##, ###).
  - Cada seção torna-se um chunk (ou é dividida em parágrafos se exceder o tamanho).
  - Retorna lista de strings com tamanho > 10 caracteres.

- `build_index(docs)`:
  - Cria `TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1,2), max_df=0.9, min_df=1)`.
  - Ajusta o vetorizador nos chunks.
  - Salva em `chatbot/index/`:
    - `vectors.npy`: matriz de vetores TF-IDF (numpy array).
    - `chunks.json`: lista dos textos dos chunks.
    - `metadados.json`: lista de dicionários com `source`, `title`, `path` de cada chunk.
    - `vectorizer.json`: configuração do vetorizador (vocabulário, idf, max_df, min_df, ngram_range, lowercase).

- `load_index()`:
  - Carrega `vectors.npy`, `chunks.json`, `metadados.json` dos arquivos salvos.
  - Retorna `(vectors, chunks, metadados)`.

**Razão do chunking por cabeçalhos:** manter cada seção de palestrante (nome, data, horário, local, tema) coesa no mesmo chunk, evitando mistura de informações entre palestrantes diferentes.

---

### 4.2 `chatbot/scrape.py` — Captura automática do conteúdo offline

**Propósito:** Baixar páginas do site oficial, converter HTML para Markdown, salvar em `offline-content/` e reindexar.

**Opções de linha de comando:**
- `--all`: páginas principais + perfis individuais de palestrantes.
- `--perfis`: apenas perfis individuais.
- `--limite N`: limita captura de perfis a N.
- `--so-reindexar`: apenas reexecuta `ingest.py` sem baixar nada.

**Fluxo:**
1. Baixa as 13 páginas principais: index, faq, palestrantes, o-festival-e-seu, ficiencias, itaipu-parquetec, sapiens, summit-tour, fims, rme, termos-de-uso, politica-de-cookies, termos-de-participacao.
2. Extrai links de perfis individuais da página de palestrantes (URLs contendo `lel_yazo_profile=`).
3. Converte HTML para Markdown com `html2text` + `BeautifulSoup` (remove scripts, styles, iframes; preserva links e imagens).
4. Aguarda 0,5s entre requisições (respeito ao servidor).
5. Reindexa com `ingest.py`.

**Nomeação de arquivos de perfil:** `palestrante-perfil-{numero_do_perfil}.md` (extraído da URL `lel_yazo_profile=NNN`).

---

### 4.3 `chatbot/chatbot.py` — Versão CLI

**Propósito:** Executável principal da versão de linha de comando. Interface interativa em português do Brasil.

**Fluxo principal:**
1. Garante que o índice existe (cria via `ingest.py` se necessário).
2. Carrega vetores, chunks, metadados e vectorizer.
3. Pergunta ao usuário: "1. Modelo local via Ollama" ou "2. Groq (nuvem)".
4. Se Ollama: inicia `ollama serve` se não estiver rodando, lista modelos disponíveis, permite seleção (ou download de novo).
5. Se Groq: requer variável de ambiente `GROQ_API_KEY`.
6. Loop de perguntas:
   - Recupera chunks via `retrieve()` (top_k=15, threshold=0.02 de similaridade de cosseno).
   - Se a pergunta menciona "palestra(s)" + mundo específico (ex: "palestras do mundo Itaipu Parquetec"), busca diretamente nos chunks de palestrantes daquele mundo via `buscar_palestras_por_mundo()` e coloca no topo do contexto.
   - Monta contexto com `format_context()`.
   - Envia ao LLM com SYSTEM_PROMPT + RAG_PROMPT.
   - Limpa a resposta com `clean_response()`.
   - Exibe com `wrap_text()` e `print_paginated()`.

**Funções centralizadas (usadas também pela web):**

- `extrair_mundo_da_pergunta(question)`: retorna o nome do mundo mencionado (ex: "Mundo Itaipu Parquetec") baseado no mapeamento `MUNDOS_NOMES`.

- `is_fonte_palestrante(fonte)`: retorna True se `source == "palestrantes.md"` ou começa com `"palestrante-perfil"`.

- `buscar_palestras_por_mundo(mundo, chunks, metadados, limit=200)`:
  - Busca chunks de palestrantes que mencionam o mundo e têm data/horário/local/bloco/espaço/sala.
  - Monta a atividade completa juntando chunks adjacentes do mesmo perfil de palestrante.
  - Para chunks seguintes, inclui linhas que começam com `### ` (tema da atividade) mas para em `## Nome` (próximo palestrante).
  - Retorna lista de dicionários com `score`, `chunk`, `source`, `title`.

- `format_answer(resultados, listagem_palestras=False, html=False)`:
  - Formata resposta a partir dos chunks.
  - Se `listagem_palestras=True`, chama `_formatar_lista_palestras(resultados, html=html)`.
  - Ordem de prioridade: palestrantes > mundos > faq > outros.
  - Para perguntas sobre pessoa específica, prioriza o chunk que menciona o nome.
  - Limpa imagens, links, quebras de linha excessivas.

- `_formatar_lista_palestras(resultados, html=False)`:
  - Formata listagem de palestras por mundo.
  - Extrai: nome do palestrante, data (`**DD/MM/AAAA**`), horário (`**HH:MM** – **HH:MM**`), local (inclui mundo), tema (linha `### `).
  - Monta linha: `**Nome** | DD/MM/AAAA | HH:MM – HH:MM | Local · Mundo X | — Tema`.
  - Se `html=True`, usa `<bold>Nome</bold>`; se `html=False`, usa ANSI `\033[1mNome\033[0m`.
  - Eventos separados por `\n\n`.

- `ollama_chat(model, mensagens)`:
  - Consulta `http://localhost:11434/api/chat` (POST).
  - Payload: `{"model": model, "messages": mensagens, "stream": False, "options": {"temperature": 0.3, "num_predict": 2000}}`.
  - Retorna o conteúdo da resposta (ou campo `thinking` se `content` estiver vazio, para modelos Qwen3.5).
  - Remove blocos "Thinking Process".

- `groq_chat(client, mensagens)`:
  - Consulta Groq com modelo `qwen/qwen3.8-27b`, temperatura 0.3, max_tokens 800.

- `clean_response(text)`:
  - Remove rótulos repetitivos ("Pergunta do usuário:", "Resposta ao usuário:", etc.).
  - Remove sequências de caracteres repetidos (loops de tokens).
  - Corrige alucinações de data ("2dezena" → "2026", "2de" → "2026").
  - Corrige erros ortográficos comuns ("dispõo" → "disponho").
  - Remove blocos "Thinking Process:" vazados.
  - Remove linhas de instruções vazadas pelo modelo.
  - Remove linhas duplicadas consecutivas.
  - Remove parágrafos duplicados ou muito similares (>= 85% de similaridade).

- `retrieve(question, vectors, chunks, metadados, vectorizer, top_k)`:
  - Converte pergunta em vetor TF-IDF.
  - Calcula similaridade de cosseno com cada chunk.
  - Retorna top_k resultados acima do threshold (0.02).

- `format_context(results)`:
  - Monta string de contexto para o LLM no formato `[Fonte: Título (arquivo)]\nchunk\n---\n`.

---

### 4.4 `web/app.py` — Versão web (Flask)

**Propósito:** Servidor Flask que expõe a interface web e a API REST do chatbot.

**Fluxo principal:**
1. Carrega o índice na inicialização (`load_rag()`).
2. Auto-seleciona o primeiro modelo Ollama disponível se o servidor estiver rodando.
3. Define rotas da API.
4. Executa servidor em host/porta configurados.

**Rotas da API:**

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Página principal do chat |
| `/api/chat` | POST | Recebe `{message}`, chama `rag_answer()`, retorna `{answer}` |
| `/api/ollama/status` | GET | Retorna `{running, models, selected, host, default_model}` |
| `/api/ollama/select` | POST | Seleciona modelo (ou `"model": ""` para desativar) |
| `/api/ollama/start` | POST | Tenta iniciar `ollama serve` |
| `/health` | GET | Health check (`{"status": "ok"}`) |

**Linha de comando:**
```bash
python3 app.py --modelo gemma3:4b --host 0.0.0.0 --porta 5000
```
Use `--modelo nenhum` para desativar o LLM local.

**`rag_answer(question)`:**
- Recupera chunks via `retrieve()` (top_k=5).
- Aplica regras de prioridade por palavra-chave.
- Se pergunta menciona palestra(s) + mundo, busca palestras daquele mundo e coloca no topo.
- Se `listagem_palestras=True`, pula o LLM e retorna diretamente o formato estruturado.
- Caso contrário, usa Ollama com SYSTEM_PROMPT + RAG_PROMPT, com fallback para chunks brutos se falhar.

**Regras de prioridade na busca:**

| Palavra-chave na pergunta | Fontes priorizadas |
|---------------------------|--------------------|
| nome próprio (ex: Cesar Brod) | palestrantes.md primeiro |
| mundo / mundos | mundos.md e index.md primeiro |
| acessibilidade / acessível / libras / acompanhamento | faq.md primeiro |
| gratuito / ingresso / inscrição / inscricao | faq.md e termos-de-participacao.md primeiro |
| quando / onde / data / local | index.md e faq.md primeiro |

---

### 4.5 `web/static/app.js` — Frontend da versão web

**Propósito:** Gerenciar a interatividade do chat no navegador.

**Responsabilidades:**
- Envio de perguntas via `fetch` para `POST /api/chat`.
- Exibição de mensagens de usuário e bot com avatares (`▲` para bot, `U` para usuário).
- Indicador de digitação (typing indicator).
- Perguntas sugeridas clicáveis na barra lateral.
- Carregamento do status do Ollama ao iniciar (`GET /api/ollama/status`).
- Seletor de modelo na barra lateral (`POST /api/ollama/select`).
- Conversão de `<bold>Nome</bold>` (enviado pelo backend) em `<strong>Nome</strong>` via função `formatarNegrito()`.
- Escape de HTML para evitar XSS.
- Paginação automática com scroll para baixo.

---

### 4.6 `web/templates/index.html` — Página principal

**Layout BrodTec (look and feel do site brodertec.com):**

- **Hero:** dados do festival (datas, local, ingresso gratuito), botão "Ir para o chat", link para site oficial.
- **Info cards:** datas, local, ingresso, mundos (seis espaços temáticos).
- **Sidebar esquerda (desktop):**
  - Sobre o evento (local, datas, ingresso, mundos, palestrantes).
  - Perguntas sugeridas clicáveis.
  - Card "Modelo de linguagem (local)": status do Ollama, modelos disponíveis, modelo ativo, seletor de modelo.
- **Área do chat:** mensagens de usuário e bot, input de texto, botão enviar, indicador de digitação.
- **Footer:** licença GPLv3+, links para site oficial, FAQ, termos de uso, BrodTec.

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

## 7. Formatação de resposta (listagem de palestras por mundo)

Quando a pergunta menciona "palestra(s)" + um mundo, o sistema lista todas as atividades daquele mundo no formato:

```
**Nome do Palestrante** | DD/MM/AAAA | HH:MM – HH:MM | Local · Mundo X | — Tema da atividade
```

- **Web:** negrito via `<bold>Nome</bold>` → convertido para `<strong>` no frontend.
- **CLI:** negrito via códigos ANSI `\033[1mNome\033[0m`.
- Cada evento separado por linha em branco (`\n\n`).

Exemplo esperado:
```
**Cesar Brod** | 27/11/2026 | 16:00 – 16:59 | Bloco 9 - Espaço 1 - Sala 2 · Mundo Itaipu Parquetec | — O mundo do desenvolvimento de sistemas utilizando IA. Limite? Cuidados? Riscos?

**Felipe Gonzalez** | 25/11/2026 | 15:00 – 15:59 | Florestan I - (Bloco 13 - Sala 1 - Espaço 1) · Mundo Itaipu Parquetec | — A importância da inovação social em territórios periféricos
```

---

## 8. Como executar (resumo)

```bash
# 1. Preparar ambiente
python3 -m venv .venv
source .venv/bin/activate
pip install groq numpy scikit-learn requests beautifulsoup4 html2text flask

# 2. (Opcional) Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b

# 3. Reindexar conteúdo offline (se já existir)
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

## 9. Observações importantes

- O conteúdo do festival é 100% offline. O chatbot **não** consulta a internet em tempo de execução.
- O índice TF-IDF é gerado automaticamente na primeira execução ou com `chatbot/scrape.py --so-reindexar`.
- Evite o modelo `qwen3.5:4b` no Ollama (modo thinking integrado deixa resposta vazia).
- Modelos recomendados: `gemma3:4b`, `phi3:mini`, `llama3.1:8b`.
- Projeto educacional sob licença GPLv3+.
- Para informações oficiais: https://festivaliguassuinova.com.br/

---

## 10. Critérios de aceite (o que deve funcionar após o vibecode)

1. `python3 chatbot/ingest.py` cria o índice em `chatbot/index/` com vetores, chunks e metadados.
2. `python3 chatbot/chatbot.py` roda no terminal, permite escolher Ollama/Groq e responde perguntas sobre o festival.
3. `python3 web/app.py --modelo gemma3:4b` sobe o servidor Flask em `http://localhost:5000`.
4. Pergunta "Quais são as palestras do mundo Itaipu Parquetec?" retorna múltiplas palestras (incluindo Cesar Brod e Felipe Gonzalez) com nome, data, horário, local e tema.
5. Nomes de palestrantes aparecem em negrito (ANSI no CLI, `<strong>` na web).
6. Cada evento aparece em linha separada.
7. O card "Modelo de linguagem (local)" na web mostra se o Ollama está online e qual modelo está ativo.
