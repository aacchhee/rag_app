## Request flow (Ask button)
```mermaid
flowchart TD
  A[User in browser<br/>types question + chooses extra_mode] --> B[Frontend JS<br/>POST /ask (JSON)]
  B --> C[Nginx (HTTPS)<br/>proxy /ask]
  C --> D[Gunicorn<br/>unix:/run/rag/rag.sock]
  D --> E[Flask: /ask handler<br/>parse JSON + defaults]

  E --> F[Embed query<br/>POST /v1/embeddings]
  F --> G[FAISS vector search<br/>top-k chunks]
  G --> H[Build source_blocks + sources[]]

  H --> I[Pass 1: Notes-first prompt<br/>question + retrieved chunks]
  I --> J[LLM chat completions<br/>POST /v1/chat/completions]
  J --> K[Sanitize + extract COVERAGE<br/>remove <think> etc.]
  K --> L[coverage = model_coverage<br/>or retrieval_coverage]

  L --> M{Do pass 2? <br/>include_extra & extra_mode}
  M -->|never| R[Skip extra]
  M -->|always| N[Pass 2: Extra prompt<br/>question + notes answer]
  M -->|auto & coverage!=full| N

  N --> O[LLM chat completions<br/>POST /v1/chat/completions]
  O --> P[Sanitize extra<br/>ensure header]
  R --> Q[Build JSON response<br/>answer_notes, coverage, sources]
  P --> Q

  Q --> S[Frontend renders:<br/>notes answer + sources<br/>extra section (optional)]
  S --> T[MathJax typeset<br/>(if enabled)]
```

