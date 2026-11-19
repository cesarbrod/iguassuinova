# Chatbot Festival Iguassu Inova 2026

Projeto educacional de chatbot RAG (Retrieval-Augmented Generation) que responde perguntas sobre o Festival Iguassu Inova 2026 usando conteúdo 100% offline do site oficial (https://festivaliguassuinova.com.br/).

## Documentação relacionada

- [`tutorial.md`](tutorial.md) — Guia detalhado de instalação, detalhamento de cada programa, razão das escolhas técnicas e explicação de TF-IDF em linguagem simples.
- [`specs.md`](specs.md) — Especificações completas para reproduzir (vibecode) este projeto em outra máquina Linux, incluindo ações necessárias do usuário, estrutura de pastas, prompts do sistema e critérios de aceite. Leia e edite esse arquivo em sua máquina local antes de proceder com as instruções! No opencode, com o modelo Nemotron 3.5 Super VL Free, use o prompt "Leia as instruções em specs.md e crie o sistema conforme as especififações." Caso teste com outras ferramentas de vibecode e modelos, fale sobre seus resultados.
- [`web/README.md`](web/README.md) — Documentação específica da versão web (Flask + HTML/CSS/JS).

## Aviso sobre o caráter educacional

Este projeto foi desenvolvido com fins **estritamente educacionais**. O objetivo é demonstrar como construir um sistema RAG (Retrieval-Augmented Generation) simples usando:

- Conteúdo capturado offline de um site real (web scraping estático)
- Busca semântica por similaridade (TF-IDF)
- Modelos de linguagem (LLM) locais via Ollama ou nuvem via Groq
- Interface de terminal em português do Brasil

O conteúdo do festival (textos, imagens, marcas, logotipos) é de propriedade do Festival Iguassu Inova e da Itaipu Parquetec. Este projeto apenas armazena cópias offline para fins de estudo e consulta local.

---

## Licença

Este software é distribuído sob os termos da **Licença Pública Geral GNU (GPL) versão 3 ou posterior**.

Você é livre para:

- Usar o software para qualquer propósito
- Estudar como ele funciona e modificá-lo
- Distribuir cópias do software original ou modificado
- Distribuir versões modificadas do software

Sob a condição de que qualquer obra derivada também seja distribuída sob a GPLv3 ou posterior, e que o código-fonte seja disponibilizado.

Veja o arquivo [`LICENSE`](LICENSE) ou https://www.gnu.org/licenses/gpl-3.0.html para o texto completo da licença.

---

## Estrutura do projeto

```
iguassuinova/
├── README.md                      # Documentação principal (este arquivo)
├── tutorial.md                    # Guia detalhado de instalação, programas e escolhas técnicas
├── specs.md                       # Especificações para reproduzir (vibecode) em outra máquina
├── LICENSE                        # Licença GPLv3+
├── .gitignore                     # Ignora __pycache__, *.pyc, index/, .env
├── offline-content/          # Conteúdo offline do site (arquivos markdown)
│   ├── index.md              # Página principal (datas, local, palestrantes, patrocinadores)
│   ├── faq.md                # Perguntas frequentes (ingressos, credenciamento, acessibilidade, LGPD)
│   ├── palestrantes.md       # Palestrantes confirmados com biografias
│   ├── palestrante-perfil-*.md  # Perfis individuais de palestrantes (capturados do site)
│   ├── mundos.md             # 6 mundos com descrições detalhadas
│   ├── ficiencias.md         # Mundo FICiências
│   ├── itaipu-parquetec.md   # Mundo Itaipu Parquetec
│   ├── sapiens.md            # Mundo Sapiens
│   ├── summit-tour.md        # Mundo Summit Tour
│   ├── fims.md               # Mundo FIMS
│   ├── rme.md                # Mundo RME
│   ├── o-festival-e-seu.md   # Chamada "O Festival é Seu"
│   ├── termos-de-uso.md      # Termos de uso
│   ├── politica-de-cookies.md# Política de cookies
│   └── termos-de-participacao.md # Termos de participação
├── chatbot/
│   ├── chatbot.py            # Executável principal (CLI; Ollama local ou Groq)
│   ├── ingest.py             # Cria índice de busca TF-IDF dos conteúdos
│   └── scrape.py             # Captura páginas do site oficial e reindexa
│   └── index/                # Índice gerado automaticamente (vectors.npy, chunks.json, etc.)
└── web/                      # Versão web (Flask + HTML/CSS/JS)
    ├── app.py                # Servidor Flask e lógica RAG + LLM local (Ollama)
    ├── README.md             # Documentação específica da versão web
    ├── static/
    │   ├── app.js            # Frontend: chat, status do Ollama, seletor de modelo
    │   └── style.css         # Estilos BrodTec
    └── templates/
        └── index.html        # Página principal com chat integrado
```

---

## Conteúdo offline

Todo o conteúdo do site https://festivaliguassuinova.com.br/ foi capturado e salvo em `offline-content/`:

Os arquivos principais são:

| Arquivo | Conteúdo |
|---------|----------|
| `index.md` | Página principal: datas (25 a 28 de novembro de 2026), local (Itaipu Parquetec, Foz do Iguaçu/PR), palestrantes em destaque, mundos, patrocinadores |
| `faq.md` | Perguntas frequentes: ingressos, credenciamento, programação, acessibilidade, LGPD, certificado |
| `palestrantes.md` | Palestrantes confirmados com biografias, incluindo Cesar Brod (Tecnologias do Futuro, 27/11/2026, 16h, Bloco 9 - Espaço 1 - Sala 2) |
| `mundos.md` | Consolidação dos 6 mundos com descrições detalhadas |
| `ficiencias.md` | Mundo FICiências: curiosidade científica e ideias jovens |
| `itaipu-parquetec.md` | Mundo Itaipu Parquetec: inovação aplicada, ciência e negócios |
| `sapiens.md` | Mundo Sapiens: conhecimento vivo, pesquisa e sociedade |
| `summit-tour.md` | Mundo Summit Tour: futuro do turismo |
| `fims.md` | Mundo FIMS: música, Feira Internacional da Música do Sul |
| `rme.md` | Mundo RME: empreendedorismo feminino, Rede Mulher Empreendedora |
| `o-festival-e-seu.md` | Chamada "O Festival é Seu" (palco, oficinas, startups) |
| `termos-de-uso.md` | Termos e condições gerais de uso |
| `politica-de-cookies.md` | Política de cookies |
| `termos-de-participacao.md` | Termos e condições de participação |

Além desses, há arquivos `palestrante-perfil-*.md` com os perfis individuais de cada palestrante (capturados do site oficial via script `scrape.py`), contendo data, horário, local e tema de cada atividade.

---

## Como fazer o setup inicial

### 1. Preparar o ambiente

Certifique-se de ter o Python 3 instalado:

```bash
python3 --version
```

Recomenda-se usar um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências Python necessárias:

```bash
pip install groq numpy scikit-learn requests beautifulsoup4 html2text flask
```

> Em sistemas onde o gerenciador de pacotes impede instalação global (ex: Ubuntu/Debian recentes), use `pip install --break-system-packages ...` ou mantenha o venv ativado.

### 2. (Opcional) Instalar o Ollama para modelo local

Para usar o LLM local totalmente offline:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b
# ou: ollama pull phi3:mini
```

> ⚠️ Evite o modelo `qwen3.5:4b`: possui modo "thinking" integrado que não pode ser desabilitado via API do Ollama, deixando a resposta vazia em RAG.

### 3. (Opcional) Configurar a chave da Groq (nuvem)

```bash
export GROQ_API_KEY=sua_chave_aqui
```

---

### 4. Capturar ou reindexar o conteúdo offline

O conteúdo já está incluído no repositório em `offline-content/`, mas você pode atualizá-lo a qualquer momento com o scraper automático:

```bash
# Captura apenas as páginas principais (rápido)
python3 chatbot/scrape.py

# Captura tudo, incluindo perfis individuais de palestrantes (demorado)
python3 chatbot/scrape.py --all

# Limita a captura de perfis (ex: 50)
python3 chatbot/scrape.py --all --limite 50

# Apenas reindexa os arquivos já existentes (sem baixar nada)
python3 chatbot/scrape.py --so-reindexar
```

O scraper baixa páginas do site oficial, converte HTML para Markdown e recria o índice TF-IDF automaticamente.

---


## Versão web (servidor local)

Além da versão CLI (`chatbot.py`), o projeto inclui uma versão web com servidor Flask e interface com look and feel BrodTec (baseado em brodertec.com).

### Como executar a versão web

```bash
cd /home/brod/scripts/opencode/iguassuinova/web
python3 app.py
```

Acesse: http://localhost:5000

### Recursos da versão web

- Interface responsiva com tema BrodTec (cores laranja, tipografia Space Grotesk/Work Sans/JetBrains Mono)
- Chat interativo com mensagens de usuário e bot
- Perguntas sugeridas clicáveis
- Paginação de mensagens (Tecle espaço para continuar)
- Word wrapping automático
- Busca baseada em conteúdo offline (sem depender de modelo LLM externo)

Para detalhes completos da versão web (instalação, escolha de modelo, rotas da API e interface), consulte [`web/README.md`](web/README.md) e [`specs.md`](specs.md).

---

### 3. Escolher o backend de LLM (apenas para versão CLI)

O chatbot suporta duas opções:

#### Opção A: Modelo local via Ollama (recomendado para uso totalmente offline)

**Pré-requisitos:**
- Ollama instalado no sistema
- Um modelo de linguagem baixado

**Passos:**

1. Inicie o servidor Ollama:
   ```bash
   ollama serve
   ```

2. Execute o chatbot:
   ```bash
   python3 chatbot.py
   ```

3. Digite **1** para modelo local.
4. Selecione o modelo disponível ou digite `0` para baixar um novo.

#### Opção B: Groq (nuvem, requer API key)

1. Obtenha uma chave de API em https://console.groq.com/
2. Defina a variável de ambiente:
   ```bash
   export GROQ_API_KEY=sua_chave_aqui
   ```
3. Execute o chatbot:
   ```bash
   python3 chatbot.py
   ```
4. Digite **2** para Groq.

---

## ⚠️ Importante: escolha de modelos locais (Ollama)

### Modelos com modo "thinking" integrado NÃO são recomendados

Modelos como o **`qwen3.5:4b`** possuem um modo de "thinking" (processo de raciocínio visível) integrado que **não pode ser desabilitado via API do Ollama**. Esse modo coloca todo o raciocínio no campo `thinking` e deixa o campo `content` vazio, impedindo que o chatbot retorne uma resposta adequada quando usado com RAG (Retrieval-Augmented Generation).

**Consequência:** Ao selecionar o `qwen3.5:4b`, o chatbot exibirá um aviso e, provavelmente, não conseguirá gerar uma resposta útil para suas perguntas.

### Modelos recomendados para uso com RAG via Ollama

| Modelo | Tamanho | Qualidade | Velocidade | Com RAG via Ollama |
|--------|---------|-----------|------------|---------------------|
| `phi3:mini` | ~2.2 GB | Boa | Rápida | ✅ Sim |
| `llama3.1:8b` | ~4.7 GB | Muito boa | Média | ✅ Sim (recomendado) |
| `gemma3:4b` | ~2.6 GB | Boa | Rápida | ✅ Sim |
| `qwen3.5:4b` | ~3.3 GB | Muito boa | Média | ❌ Não (modo thinking integrado) |

**Recomendação:** Use `phi3:mini` (leve e rápido) ou `llama3.1:8b` (melhor qualidade). Para baixar:

```bash
ollama pull phi3:mini
# ou
ollama pull llama3.1:8b
```

---

## Melhorias de formatação

O chatbot inclui as seguintes melhorias de exibição no terminal:

- **Word wrapping automático**: as respostas são ajustadas à largura do terminal sem cortar palavras ao final da linha
- **Negrito nos rótulos**: "Você:" e "Bot:" são exibidos em negrito (códigos ANSI)
- **Sem rótulos repetitivos**: não são incluídos "Pergunta do usuário:" ou "Resposta ao usuário:" nas respostas
- **Paginação**: quando a resposta excede a altura do terminal, exibe "Tecle espaço para continuar..." para avançar página por página
- **Proteção contra loops de tokens**: detecta e remove sequências de caracteres repetidos geradas pelo modelo
- **Remoção de parágrafos duplicados**: elimina parágrafos repetidos ou muito similares (>= 85% de similaridade)
- **Correção de alucinações de data**: corrige padrões como "2dezena" ou "2de" para "2026"
- **Correção ortográfica**: corrige erros comuns de alucinação do modelo (ex: "dispõo" → "disponho")
- **Remoção de instruções vazadas**: remove linhas de instrução que o modelo às vezes inclui na resposta
- **Modo não-interativo**: ao executar via pipe/redirecionamento, a paginação é desabilitada automaticamente

---

## Exemplos de perguntas

- "Quais são os mundos do evento?"
- "Qual é o tema do Mundo RME?"
- "Quando e onde acontece o festival?"
- "A entrada é gratuita?"
- "Quais são os shows nacionais?"
- "Qual será a palestra de Cesar Brod?"
- "Qual o dia, horário e local da palestra de Cesar Brod?"
- "Preciso me inscrever nas oficinas?"
- "Haverá certificado de participação?"
- "Quais recursos de acessibilidade estarão disponíveis?"

Digite `sair`, `exit` ou `quit` para encerrar o chatbot.

---

## Documentação detalhada

- [`tutorial.md`](tutorial.md): guia completo de instalação, detalhamento de cada programa (`ingest.py`, `scrape.py`, `chatbot.py`, `app.py`, `app.js`), razão das escolhas técnicas e explicação de TF-IDF em linguagem simples.
- [`specs.md`](specs.md): especificações para reproduzir (vibecode) o projeto em outra máquina Linux, incluindo ações do usuário, estrutura de pastas, prompts do sistema, mapeamento de mundos, regras de prioridade na busca e critérios de aceite.

---

## Responsabilidade

Este projeto é fornecido "como está", sem garantias de qualquer tipo. O conteúdo do festival está sujeito a alterações pela organização. Para informações oficiais e atualizadas, consulte sempre o site oficial: https://festivaliguassuinova.com.br/

---

## Contribuição

Este é um projeto educacional. Contribuições são bem-vindas, desde que respeitem a licença GPLv3+ e o caráter educacional do projeto.
