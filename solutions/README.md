# Soluções propostas

> Os resultados abaixo registram um experimento local anterior. Os dados e o
> modelo pessoal não são distribuídos; consulte
> [`DATA_POLICY.md`](../DATA_POLICY.md).

As abordagens compartilham exatamente o mesmo dataset, representação de
posição, divisões cronológicas e métricas. Somente a estratégia usada para
pontuar os lances legais muda. Isso permite uma comparação justa e evita manter
três pipelines divergentes.

| Solução | Papel | Teste Top-1 | Situação |
|---|---|---:|---|
| [Frequência](frequency/README.md) | baseline mínimo | 13,21% | avaliada |
| [Regressão logística](logistic/README.md) | modelo linear | 6,15% | avaliada |
| [Gradient boosting](hist_gradient_boosting/README.md) | árvores não lineares | 31,13% | selecionada |
| [Maia Chess](maia/README.md) | transferência de aprendizado | não medido | exploratória |

Após a seleção pela validação, o gradient boosting foi retreinado com treino e
validação e atingiu 31,61% Top-1 no teste isolado. Um novo treinamento local
grava o vencedor em `artifacts/final_model.joblib`; esse arquivo é ignorado pelo
Git.

## Código

Cada solução ativa possui seu próprio pacote:

```text
src/botchess/solutions/
├── base.py                         contrato e persistência dos modelos
├── registry.py                     catálogo consumido pelo treinamento
├── frequency/model.py              baseline
├── logistic/model.py               regressão logística
└── hist_gradient_boosting/model.py boosting selecionado
```

Para acrescentar outra abordagem, crie um pacote em
`src/botchess/solutions/` e registre uma `SolutionSpec` em `registry.py`. Modelos
baseados em pares posição-lance reutilizam a matriz de candidatos construída
uma única vez pelo orquestrador.

## Políticas usadas durante a partida

**Imitador** e **Coerente** não são modelos de treinamento adicionais. Os dois
usam o artefato informado na execução; o segundo apenas filtra erros táticos rasos
antes de escolher pelo ranking pessoal. Por isso, essas políticas ficam em
`src/botchess/play_modes.py`, fora do catálogo comparativo de soluções.
