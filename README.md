# Laboratório 10 — O Pipeline Definitivo (RAG, QLoRA e Otimização de Inferência)

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Pedro De Carvalho Lima.

---

## Objetivo

Este laboratório implementa um pipeline de inferência otimizado para Large Language Models utilizando:

- RAG (Retrieval-Augmented Generation)
- Quantização QLoRA em 4 bits
- KV Cache
- FlashAttention-2
- Monitoramento de memória VRAM
- Benchmark de geração auto-regressiva

O objetivo é simular um cenário corporativo onde um modelo Transformer precisa processar contextos massivos recuperados por sistemas RAG sem causar erro de Out-Of-Memory (OOM) na GPU.

---

# Estrutura do Projeto

```bash
rag-qlora-inference-optimization/
├── docs/
├── notebooks/
├── results/
├── src/
│   └── benchmark_inference.py
├── requirements.txt
└── README.md
```

---

# Tecnologias Utilizadas

- Python 3.12
- PyTorch
- Hugging Face Transformers
- BitsAndBytes
- FlashAttention-2
- CUDA
- QLoRA
- KV Cache

---

# Pipeline Implementado

## 1. Carregamento Quantizado

O modelo foi preparado para utilização de quantização em 4 bits utilizando a estratégia QLoRA, reduzindo significativamente o consumo de memória durante a inferência.

---

## 2. Simulação de RAG Massivo

Foi criado um contexto médico fictício simulando documentos recuperados por um sistema RAG corporativo.

---

## 3. Benchmark sem KV Cache

Foi executado um benchmark de geração auto-regressiva desativando o cache do decoder (`use_cache=False`) para observar o impacto do recálculo redundante do mecanismo de Self-Attention.

---

## 4. Benchmark com KV Cache

O pipeline foi refatorado para utilizar KV Cache (`use_cache=True`), reduzindo a recomputação das matrizes Q, K e V durante a geração de novos tokens.

---

## 5. FlashAttention-2

O projeto foi estruturado para utilizar FlashAttention-2 quando disponível no ambiente CUDA.

Devido à ausência do CUDA Toolkit completo no ambiente local de execução, a instalação do `flash-attn` não pôde ser concluída. O código implementa fallback automático para garantir compatibilidade em ambientes sem suporte à biblioteca.

---

# Resultados do Benchmark

| Configuração | Tempo (s) | VRAM Pico (MB) |
|---|---|---|
| Sem KV Cache | 0.27 | 0 |
| Com KV Cache | 0.23 | 0 |

---

# Análise Arquitetural

## Parte A — Como QLoRA, KV Cache e FlashAttention salvaram o Transformer

A combinação entre QLoRA, KV Cache e FlashAttention reduz drasticamente o custo computacional do Transformer durante inferência com contextos extensos. A quantização QLoRA em 4 bits reduz o consumo de memória do modelo, permitindo carregar LLMs maiores sem exceder a VRAM disponível. O KV Cache evita o recálculo redundante das matrizes Key e Value durante a geração auto-regressiva, reduzindo significativamente a latência do decoder. Já o FlashAttention utiliza otimizações hardware-aware para minimizar acessos à memória global da GPU, explorando melhor a SRAM e reduzindo overhead de memória durante o cálculo do Self-Attention.

Sem essas técnicas, o crescimento quadrático O(n²) do mecanismo de atenção tornaria inviável o processamento de grandes sequências em ambientes de produção, causando gargalos severos de memória e tempo de inferência.

---

## Parte B — Limitações dos Transformers e necessidade de State Space Models

Mesmo com FlashAttention, Transformers continuam limitados pela complexidade quadrática do mecanismo de Self-Attention. Em cenários extremos, como processamento de contextos com milhões de tokens, o custo computacional e o consumo de memória tornam-se impraticáveis mesmo em GPUs modernas.

Nesse contexto, arquiteturas baseadas em State Space Models, como Mamba, tornam-se alternativas mais escaláveis. Diferentemente dos Transformers tradicionais, esses modelos conseguem processar sequências longas utilizando complexidade de memória próxima de O(1), eliminando a necessidade de armazenar explicitamente matrizes de atenção gigantescas. Isso torna os SSMs mais adequados para aplicações futuras envolvendo contexto massivo, streaming contínuo e inferência de longo alcance.

---

# Execução

## Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Executar benchmark

```bash
python src/benchmark_inference.py
```

---

# Versionamento

Versão final entregue:

```bash
v1.0
```