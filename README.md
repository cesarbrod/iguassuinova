# Chatbot Festival Iguassu Inova 2026

Projeto educacional de chatbot RAG (Retrieval-Augmented Generation) que responde perguntas sobre o Festival Iguassu Inova 2026 usando conteúdo 100% offline do site oficial (https://festivaliguassuinova.com.br/).

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

Veja o arquivo `LICENSE` (ou https://www.gnu.org/licenses/gpl-3.0.html) para o texto completo da licença.

---

## Estrutura do projeto

```
iguassuinova/
├── README.md                      # Documentação principal (este arquivo)
├── offline-content/          # Conteúdo offline do site (14 arquivos markdown)
│   ├── index.md              # Página principal (datas, local, palestrantes, patrocinadores)
│   ├── faq.md                # Perguntas frequentes (ingressos, credenciamento, acessibilidade, LGPD)
│   ├── palestrantes.md       # 223 palestrantes confirmados com biografias
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
│   ├── chatbot.py            # Executável principal (pergunta local vs Groq no início)
│   ├── ingest.py             # Cria índice de busca TF-IDF dos conteúdos
│   └── index/                # Índice gerado automaticamente (vectors.npy, chunks.json, etc.)
```

---

## Conteúdo offline (14 arquivos)

Todo o conteúdo do site https://festivaliguassuinova.com.br/ foi capturado e salvo em `offline-content/`:

| Arquivo | Conteúdo |
|---------|----------|
| `index.md` | Página principal: datas (25 a 28 de novembro de 2026), local (Itaipu Parquetec, Foz do Iguaçu/PR), palestrantes em destaque, mundos, patrocinadores |
| `faq.md` | Perguntas frequentes: ingressos, credenciamento, programação, acessibilidade, LGPD, certificado |
| `palestrantes.md` | 223 palestrantes confirmados com biografias, incluindo Cesar Brod (Tecnologias do Futuro, 27/11/2026, 16h, Bloco 9 - Espaço 1 - Sala 2) |
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

---

## Como fazer o setup inicial

### 1. Preparar o ambiente

Certifique-se de ter o Python 3 instalado:

```bash
python3 --version
```

Instale as dependências Python necessárias:

```bash
pip3 install --break-system-packages groq numpy scikit-learn
```

### 2. Atualizar o conteúdo offline do site (opcional)

Se desejar atualizar o conteúdo capturado do site https://festivaliguassuinova.com.br/, siga estes passos:

#### 2.1. Capturar páginas do site

Use uma ferramenta de captura de páginas (como `webfetch`, `curl` ou `wget`) para baixar as páginas relevantes:

```bash
# Exemplo de páginas a serem capturadas:
# https://festivaliguassuinova.com.br/
# https://festivaliguassuinova.com.br/faq.html
# https://festivaliguassuinova.com.br/palestrantes.html
# https://festivaliguassuinova.com.br/o-festival-e-seu.html
# https://festivaliguassuinova.com.br/ficiencias.html
# https://festivaliguassuinova.com.br/itaipu-parquetec.html
# https://festivaliguassuinova.com.br/sapiens.html
# https://festivaliguassuinova.com.br/summit-tour.html
# https://festivaliguassuinova.com.br/fims.html
# https://festivaliguassuinova.com.br/rme.html
# https://festivaliguassuinova.com.br/termos-de-uso.html
# https://festivaliguassuinova.com.br/politica-de-cookies.html
# https://festivaliguassuinova.com.br/termos-de-participacao.html
```

Salve cada página capturada como arquivo `.md` (markdown) dentro da pasta `offline-content/`, mantendo a nomenclatura descritiva.

#### 2.2. Adicionar páginas de perfis individuais de palestrantes

O site de palestrantes (`https://festivaliguassuinova.com.br/palestrantes.html`) contém links individuais para cada palestrante no formato:

```
https://festivaliguassuinova.com.br/palestrantes/?lel_yazo_profile=NNN
```

Para adicionar informações detalhadas de um palestrante específico (ex: Cesar Brod, perfil 225), acesse a URL correspondente e salve o conteúdo em `offline-content/palestrantes.md` ou em um arquivo dedicado.

#### 2.3. Recriar o índice de busca

Após atualizar ou adicionar arquivos em `offline-content/`, recria o índice TF-IDF:

```bash
cd /home/brod/scripts/opencode/iguassuinova/chatbot
python3 ingest.py
```

Isso gerará os arquivos em `index/` (vectors.npy, chunks.json, metadados.json, vectorizer.json) usados pelo chatbot para busca por similaridade.

### 3. Escolher o backend de LLM

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

## Responsabilidade

Este projeto é fornecido "como está", sem garantias de qualquer tipo. O conteúdo do festival está sujeito a alterações pela organização. Para informações oficiais e atualizadas, consulte sempre o site oficial: https://festivaliguassuinova.com.br/

---

## Contribuição

Este é um projeto educacional. Contribuições são bem-vindas, desde que respeitem a licença GPLv3+ e o caráter educacional do projeto.