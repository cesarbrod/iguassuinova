# Tutorial — Chatbot Festival Iguassu Inova 2026

Este tutorial descreve, em detalhes, cada programa do projeto, a razão das escolhas técnicas adotadas e as instruções completas para instalar e executar todo o sistema, nas versões de linha de comando (CLI) e web.

---

## 1. Visão geral

O projeto é um chatbot educacional de RAG (Retrieval-Augmented Generation) que responde perguntas sobre o Festival Iguassu Inova 2026 com base em conteúdo 100% offline capturado do site oficial (https://festivaliguassuinova.com.br/). Não consulta a internet em tempo de execução: toda a informação vem de arquivos Markdown locais indexados por busca semântica TF-IDF.

O sistema suporta duas interfaces:

- **CLI** (`chatbot/chatbot.py`): terminal interativo em português do Brasil, com suporte a modelo local via Ollama ou nuvem via Groq.
- **Web** (`web/app.py`): servidor Flask com interface responsiva (look and feel BrodTec), suporte a modelo local via Ollama (padrão `gemma3:4b`) e seleção de modelo em tempo real pela interface.

---

## 2. Estrutura de pastas

```
iguassuinova/
├── README.md                      # Documentação principal do projeto
├── tutorial.md                    # Este arquivo: detalhamento dos programas e instalação
├── web/README.md                  # Documentação específica da versão web
├── LICENSE                        # GPLv3+
├── .gitignore                     # Ignora __pycache__, *.pyc, index/, .env
├── offline-content/               # Conteúdo offline capturado do site (Markdown)
│   ├── index.md                   # Página principal (datas, local, palestrantes, patrocinadores)
│   ├── faq.md                     # Perguntas frequentes
│   ├── palestrantes.md            # Palestrantes confirmados com biografias
│   ├── palestrante-perfil-*.md    # Perfis individuais de palestrantes (capturados via scrape.py)
│   ├── mundos.md                  # Seis mundos temáticos
│   ├── ficiencias.md              # Mundo FICiências
│   ├── itaipu-parquetec.md        # Mundo Itaipu Parquetec
│   ├── sapiens.md                 # Mundo Sapiens
│   ├── summit-tour.md             # Mundo Summit Tour
│   ├── fims.md                    # Mundo FIMS
│   ├── rme.md                     # Mundo RME
│   ├── o-festival-e-seu.md        # Chamada "O Festival é Seu"
│   ├── termos-de-uso.md
│   ├── politica-de-cookies.md
│   └── termos-de-participacao.md
├── chatbot/
│   ├── chatbot.py                 # Executável principal da versão CLI
│   ├── ingest.py                  # Cria índice de busca TF-IDF dos conteúdos
│   └── scrape.py                  # Captura páginas do site oficial e reindexa
│   └── index/                     # Índice gerado automaticamente (vectors.npy, chunks.json, metadados.json, vectorizer.json)
└── web/                           # Versão web (Flask + HTML/CSS/JS)
    ├── app.py                     # Servidor Flask e lógica RAG + LLM local (Ollama)
    ├── README.md                  # Documentação da versão web
    ├── static/
    │   ├── app.js                 # Frontend: chat, status do Ollama, seletor de modelo
    │   └── style.css              # Estilos BrodTec
    └── templates/
        └── index.html             # Página principal com chat integrado
```

---

## 3. Programas utilizados

### 3.1 `chatbot/ingest.py` — Construção do índice de busca

**O que faz:** Lê todos os arquivos Markdown de `offline-content/`, divide o texto em chunks (trechos) preservando seções de cabeçalho markdown (#, ##, ###), cria uma matriz TF-IDF e salva os artefatos em `chatbot/index/`:
- `vectors.npy`: matriz de vetores TF-IDF
- `chunks.json`: lista dos textos dos chunks
- `metadados.json`: metadados (arquivo de origem, título, caminho) de cada chunk
- `vectorizer.json`: configuração e vocabulário do vetorizador TF-IDF

**Por que TF-IDF:** É uma técnica leve, rápida, totalmente offline e não requer modelo de linguagem externo para indexar. É suficiente para busca por similaridade de palavras-chave sobre um conjunto limitado de documentos estáticos (cerca de 247 arquivos e 3.100+ chunks).

#### O que é TF-IDF (em linguagem simples)

TF-IDF significa **Term Frequency–Inverse Document Frequency** (Frequência do Termo – Frequência Inversa do Documento). Apesar do nome complicado, a ideia é bem simples:

Imagine que você tem um monte de documentos (no nosso caso, os arquivos Markdown do festival) e quer encontrar quais deles são mais relevantes para uma pergunta. O TF-IDF faz isso dando uma "nota" a cada palavra de cada documento, considerando duas coisas:

1. **TF (Frequência do Termo):** Quantas vezes a palavra aparece no documento. Se a palavra "Itaipu" aparece muitas vezes num documento, esse documento provavelmente fala sobre Itaipu.

2. **IDF (Frequência Inversa do Documento):** Quão rara é a palavra em todos os documentos. Palavras que aparecem em quase todos os documentos (como "o", "a", "e", "de", "festival") não ajudam a diferenciar um documento do outro, então recebem pouca nota. Palavras raras e específicas (como "Itaipu Parquetec", "FICiências", "Cesar Brod") recebem muita nota, porque são bons indicadores do assunto.

A nota final (TF-IDF) é alta quando uma palavra aparece **muitas vezes em um documento específico** e **pouco nos outros documentos**. Assim, quando você faz uma pergunta, o sistema compara as palavras da pergunta com as notas de cada documento e retorna aqueles com a maior pontuação — ou seja, os que provavelmente têm a resposta.

No projeto, cada trecho (chunk) de texto recebe um vetor de notas TF-IDF. A pergunta do usuário também é convertida num vetor do mesmo jeito. O sistema então calcula a similaridade entre o vetor da pergunta e os vetores dos trechos, devolvendo os trechos mais parecidos.

**Chunking consciente de seções:** O `ingest.py` divide o texto por cabeçalhos markdown (#, ##, ###) para manter cada seção de palestrante (nome, data, horário, local, tema) coesa no mesmo chunk, evitando que informação de um palestrante se misture com outro.

### 3.2 `chatbot/scrape.py` — Captura automática do conteúdo offline

**O que faz:** Baixa páginas do site oficial do festival, converte HTML para Markdown (preservando links e imagens), salva em `offline-content/` e, ao final, reexecuta `ingest.py` para recriar o índice.

**Opções de linha de comando:**
- `--all`: captura páginas principais + perfis individuais de palestrantes (equivalente a `--perfis`).
- `--perfis`: captura apenas perfis individuais (além das páginas principais se usado sozinho).
- `--limite N`: limita a captura de perfis individuais a N (útil para testes ou para evitar sobrecarregar o servidor).
- `--so-reindexar`: apenas reexecuta o `ingest.py` sobre os arquivos já existentes em `offline-content/`, sem baixar nada.

**Por que capturar o site:** O conteúdo do festival é dinâmico (novos palestrantes, horários, locais). O scraper permite atualizar a base offline sem depender de conexão com a internet em tempo de execução do chatbot.

**Respeito ao servidor:** O scraper aguarda 0,5 segundo entre requisições e usa um User-Agent identificável, evitando sobrecarga no servidor do festival.

### 3.3 `chatbot/chatbot.py` — Versão de linha de comando (CLI)

**O que faz:** É o executável principal da versão CLI. Carrega o índice TF-IDF, permite escolher o backend de LLM (Ollama local ou Groq nuvem), recupera chunks relevantes para a pergunta do usuário e gera a resposta com base no contexto offline.

**Fluxo de execução:**
1. Verifica se o índice existe; caso contrário, cria via `ingest.py`.
2. Pergunta ao usuário qual backend usar (1 = Ollama local, 2 = Groq).
3. Se Ollama: inicia o servidor (`ollama serve`) se necessário e lista os modelos disponíveis para seleção.
4. Se Groq: requer a variável de ambiente `GROQ_API_KEY`.
5. Loop de perguntas: para cada pergunta, recupera os chunks mais similares, monta o contexto e envia ao LLM selecionado.
6. A resposta é limpa (remoção de rótulos repetitivos, loops de tokens, blocos "Thinking Process" vazados, etc.) e exibida com paginação no terminal.

**Por que Ollama e Groq:** Ollama permite execução 100% offline e sem custo, ideal para uso educacional e locais sem internet. Groq oferece alternativa rápida na nuvem para quem tem API key e quer melhor qualidade de geração.

**Suporte a palestras por mundo:** Quando a pergunta menciona "palestra(s)" + um mundo específico (ex: "palestras do mundo Itaipu Parquetec"), o programa busca diretamente nos chunks de palestrantes aquele mundo, garantindo que todas as atividades daquele mundo sejam listadas, não apenas as que aparecem nos primeiros resultados de similaridade.

**Formatação no terminal:** Nomes de palestrantes em negrito via códigos ANSI (`\033[1m...\033[0m`), word wrapping automático, paginação ("Tecle espaço para continuar") e remoção de parágrafos duplicados.

### 3.4 `web/app.py` — Versão web (Flask)

**O que faz:** Servidor Flask que expõe a interface web e a API REST do chatbot. Carrega o índice TF-IDF na inicialização e, para cada pergunta enviada pelo frontend via `/api/chat`, recupera os chunks relevantes e retorna a resposta.

**Rotas da API:**
| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Página principal do chat |
| `/api/chat` | POST | Envia uma pergunta e recebe a resposta |
| `/api/ollama/status` | GET | Status do Ollama, modelos disponíveis e modelo ativo |
| `/api/ollama/select` | POST | Seleciona o modelo local a ser usado |
| `/api/ollama/start` | POST | Tenta iniciar o servidor Ollama |
| `/health` | GET | Health check |

**Escolha do modelo:** Por padrão, usa `gemma3:4b` (Ollama). Pode ser alterado via linha de comando (`--modelo`) ou pela interface web (select na barra lateral). Use `--modelo nenhum` para desativar o LLM local e responder apenas com chunks brutos.

**Por que Flask:** Flask é leve, simples de instalar e executar localmente, e não requer infraestrutura adicional. É adequado para um servidor local educacional acessível via `http://localhost:5000`.

**Não usar LLM para listagem de palestras:** Quando a pergunta é listagem de palestras por mundo, o contexto contém centenas de itens. Enviar isso ao LLM geraria fragmentos aleatórios. Nesse caso, a resposta é gerada diretamente pelo formatador estruturado (nome, data, horário, local, tema), sem passar pelo LLM.

### 3.5 `web/static/app.js` — Frontend da versão web

**O que faz:** Gerencia a interatividade do chat: envio de perguntas via fetch para `/api/chat`, exibição de mensagens de usuário e bot, indicador de digitação, perguntas sugeridas clicáveis, carregamento do status do Ollama e seletor de modelo.

**Formatação:** Converte marcações `<bold>Nome</bold>` (enviadas pelo backend) em `<strong>Nome</strong>` (negrito HTML) e escapa o restante do texto para evitar XSS.

---

## 4. Razão das escolhas técnicas

| Escolha | Razão |
|---------|-------|
| Python 3 | Linguagem amplamente disponível, fácil de manter e com boas bibliotecas para NLP e web. |
| TF-IDF (scikit-learn) | Leve, rápido, offline, sem dependência de modelos de linguagem pesados. Adequado para busca sobre documentos estáticos. |
| Ollama (modelo local) | Execução 100% offline, sem custo, sem envio de dados para terceiros. Ideal para uso educacional. |
| Groq (nuvem) | Alternativa rápida na nuvem para quem tem API key e quer melhor qualidade de geração. |
| Flask | Simples de instalar e executar localmente, sem infraestrutura adicional. |
| html2text + BeautifulSoup | Conversão robusta de HTML para Markdown, preservando links e imagens. |
| Markdown offline | Facilita versionamento, leitura humana e indexação por TF-IDF. |
| Chunking por cabeçalhos | Mantém coesas as seções de cada palestrante (nome, data, local, tema), evitando mistura de informações. |

---

## 5. Pré-requisitos

- **Python 3** (recomendado 3.9+)
- **pip** (geralmente já vem com Python)
- **Ollama** (opcional, apenas para usar modelo local): https://ollama.com/
- **Acesso à internet** apenas para a etapa de instalação e captura do conteúdo (o chatbot em si funciona 100% offline)

---

## 6. Instalação completa

### 6.1 Clonar o repositório

```bash
git clone https://github.com/cesarbrod/iguassuinova.git
cd iguassuinova
```

### 6.2 Instalar dependências Python

```bash
pip3 install --break-system-packages groq numpy scikit-learn requests beautifulsoup4 html2text flask
```

> O parâmetro `--break-system-packages` pode ser necessário em sistemas onde o gerenciador de pacotes do sistema impede instalação global (ex: Ubuntu/Debian recentes). Se estiver usando um ambiente virtual, omita esse parâmetro.

### 6.3 (Opcional) Instalar o Ollama para modelo local

Se quiser usar o LLM local (recomendado para uso totalmente offline):

```bash
# Instalar Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar o servidor
ollama serve

# Baixar um modelo recomendado (em outro terminal)
ollama pull gemma3:4b
# ou
ollama pull phi3:mini
```

> ⚠️ Evite o modelo `qwen3.5:4b`: ele possui modo "thinking" integrado que não pode ser desabilitado via API do Ollama, deixando o conteúdo da resposta vazio quando usado com RAG.

### 6.4 Capturar o conteúdo offline do site

O conteúdo já está incluído no repositório, mas você pode atualizá-lo a qualquer momento:

```bash
# Capturar apenas as páginas principais (rápido)
python3 chatbot/scrape.py

# Capturar tudo, incluindo perfis individuais de palestrantes (demorado; ~247 perfis)
python3 chatbot/scrape.py --all

# Limitar a captura de perfis (ex: 50)
python3 chatbot/scrape.py --all --limite 50

# Apenas reindexar os arquivos já existentes (sem baixar nada)
python3 chatbot/scrape.py --so-reindexar
```

Após a captura, o índice TF-IDF é recriado automaticamente em `chatbot/index/`.

---

## 7. Como executar

### 7.1 Versão CLI (terminal)

```bash
cd chatbot
python3 chatbot.py
```

Ao iniciar, escolha o backend:
- `1` — Modelo local via Ollama (totalmente offline)
- `2` — Groq (nuvem, requer `GROQ_API_KEY`)

Exemplos de perguntas:
- "Quais são os mundos do evento?"
- "Quando e onde acontece o festival?"
- "Quais são as palestras do mundo Itaipu Parquetec?"
- "Qual o dia, horário e local da palestra de Cesar Brod?"

Digite `sair`, `exit` ou `quit` para encerrar.

### 7.2 Versão web (servidor local)

```bash
cd web
python3 app.py
```

Acesse: http://localhost:5000

Opções de linha de comando:
```bash
python3 app.py --modelo gemma3:4b   # padrão
python3 app.py --modelo phi3:mini    # modelo mais leve
python3 app.py --modelo nenhum       # desativa LLM local (chunks brutos)
python3 app.py --porta 8080          # porta customizada
```

Na barra lateral do chat há um card "Modelo de linguagem (local)" que mostra se o Ollama está online, quantos modelos estão disponíveis e qual modelo está ativo. Você pode trocar o modelo a qualquer momento pelo select, sem reiniciar o servidor.

---

## 8. Exemplos de perguntas testadas

- "Quais são os mundos do evento?"
- "Quando e onde acontece o festival?"
- "A entrada é gratuita?"
- "Quais são as palestras do mundo Itaipu Parquetec?" → lista Cesar Brod, Felipe Gonzalez, Gustavo Possetti e muitos outros, com data, horário, local e tema.
- "Qual o dia, horário e local da palestra de Cesar Brod?"
- "Quais recursos de acessibilidade estarão disponíveis?"
- "Preciso me inscrever nas oficinas?"

---

## 9. Observações importantes

- O conteúdo do festival é 100% offline. O chatbot **não** consulta a internet em tempo de execução.
- O índice TF-IDF é gerado automaticamente na primeira execução ou pode ser recriado com `chatbot/scrape.py --so-reindexar`.
- As respostas usam apenas os conteúdos offline do site oficial do festival. Para informações oficiais e atualizadas, consulte sempre https://festivaliguassuinova.com.br/.
- Este é um projeto educacional distribuído sob a Licença GPLv3 ou posterior.

---

## 10. Licença

Este software é distribuído sob os termos da **Licença Pública Geral GNU (GPL) versão 3 ou posterior**. Veja o arquivo [`LICENSE`](LICENSE) para o texto completo.
