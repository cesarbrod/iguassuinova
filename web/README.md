# Versão web — Chatbot Festival Iguassu Inova 2026

Interface web local (Flask + HTML/CSS/JS) com look and feel BrodTec, integrada ao índice RAG (TF-IDF) dos conteúdos offline do festival. Pode usar um modelo de linguagem local via Ollama para gerar as respostas, funcionando totalmente offline.

## Como executar

```bash
cd /home/brod/scripts/opencode/iguassuinova/web
python3 app.py
```

Acesse: http://localhost:5000

## Escolha do modelo de linguagem (LLM local)

Por padrão, a interface usa o modelo `gemma3:4b` (Ollama) para gerar as respostas com base nos conteúdos offline.

Execute com um modelo específico:

```bash
python3 app.py --modelo gemma3:4b
```

Outras opções:

```bash
python3 app.py --modelo phi3:mini      # modelo mais leve e rápido
python3 app.py --modelo llama3.1:8b    # melhor qualidade (se disponível)
python3 app.py --modelo nenhum         # desativa o LLM local; responde direto dos conteúdos offline
```

Parâmetros disponíveis:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--modelo` | `gemma3:4b` | Modelo Ollama a usar. Use `nenhum` para desativar o LLM local. |
| `--host` | `0.0.0.0` | Host do servidor. |
| `--porta` | `5000` | Porta do servidor. |

### Pré-requisitos para o LLM local

- Ollama instalado e rodando: `ollama serve`
- Um modelo baixado, por exemplo:

```bash
ollama pull gemma3:4b
# ou
ollama pull phi3:mini
```

> ⚠️ Evite o modelo `qwen3.5:4b`: ele possui modo "thinking" integrado que não pode ser desabilitado via API do Ollama, deixando o conteúdo da resposta vazio quando usado com RAG.

## Interface

Na barra lateral do chat há um card "Modelo de linguagem (local)" que mostra:

- Se o Ollama está online (localhost:11434)
- Quantos modelos estão disponíveis
- Qual modelo está ativo (ex: `gemma3:4b`)

Você pode trocar o modelo a qualquer momento pelo select na interface, sem reiniciar o servidor. Se escolher "Nenhum", as respostas serão geradas diretamente a partir dos conteúdos offline (busca TF-IDF), sem usar LLM.

## Estrutura da versão web

```
web/
├── app.py              # Servidor Flask e lógica RAG + LLM local (Ollama)
├── README.md           # Este arquivo
├── static/
│   ├── app.js          # Frontend: chat, status do Ollama, seletor de modelo
│   └── style.css       # Estilos BrodTec
└── templates/
    └── index.html      # Página principal com chat integrado
```

## Rotas da API

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Página principal do chat |
| `/api/chat` | POST | Envia uma pergunta e recebe a resposta |
| `/api/ollama/status` | GET | Status do Ollama, modelos disponíveis e modelo ativo |
| `/api/ollama/select` | POST | Seleciona o modelo local a ser usado |
| `/api/ollama/start` | POST | Tenta iniciar o servidor Ollama |
| `/health` | GET | Health check |

## Observações

- O conteúdo do festival é 100% offline (pasta `offline-content/` na raiz do projeto).
- O índice de busca TF-IDF é gerado automaticamente na pasta `chatbot/index/` se ainda não existir.
- As respostas usam apenas os conteúdos offline do site oficial do festival (não consultam a internet).

---

Projeto educacional — Licença GPLv3+. Veja o [README raiz](../README.md) para detalhes do projeto, estrutura geral e licença.