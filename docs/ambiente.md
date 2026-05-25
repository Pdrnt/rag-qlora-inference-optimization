# Observação sobre FlashAttention-2

Durante a configuração do ambiente, a instalação da biblioteca `flash-attn` falhou porque o sistema local não possui `nvcc` disponível nem a variável `CUDA_HOME` configurada.

O código do laboratório foi estruturado para tentar usar `attn_implementation="flash_attention_2"` quando disponível. Caso o ambiente não suporte FlashAttention-2, o pipeline executa um fallback seguro com a implementação padrão do PyTorch/Transformers, registrando essa limitação no relatório de benchmark.

Essa decisão mantém o experimento reproduzível em ambientes sem CUDA Toolkit completo, sem remover a análise arquitetural exigida pelo laboratório.