# Maia Chess

Alternativa baseada em uma rede Leela Chess Zero pré-treinada para imitar
jogadores humanos. Foi a primeira abordagem estudada, mas não participa do
pipeline atual porque sua preparação e seu ambiente de treinamento são muito
mais pesados que os modelos clássicos comparados.

## Materiais de referência

- [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess): código e pesos
  publicados pelo projeto Maia;
- `../../docs/research/model-selection.md`: pesquisa que recomendou Maia e
  comparou outras arquiteturas.

O checkout upstream, seus pesos, os PGNs experimentais e os arquivos antigos de
Colab não são duplicados neste repositório. Eles são grandes, possuem ciclo e
licença próprios e não são necessários para executar a solução selecionada.

## Situação

Ainda não há uma avaliação Maia no mesmo split cronológico das soluções
clássicas. Ele só deve entrar na tabela principal depois de usar uma fonte de
dados autorizada, somente com os lances do jogador-alvo, e produzir as mesmas
métricas Top-k e MRR.
