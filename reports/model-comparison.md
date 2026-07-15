# Comparação dos modelos

> Registro histórico de um experimento local. O dataset e o modelo pessoal não
> são distribuídos; consulte [`DATA_POLICY.md`](../DATA_POLICY.md).

Resultados produzidos pelo pipeline local em divisões cronológicas. A seleção
usa validação; o conjunto de teste não participa da escolha.

| Modelo | Val. Top-1 | Val. Top-3 | Val. MRR | Teste Top-1 | Teste Top-3 | Teste MRR | ms/posição |
|---|---:|---:|---:|---:|---:|---:|---:|
| frequency | 0.1286 | 0.2550 | 0.2470 | 0.1321 | 0.2819 | 0.2583 | 0.13 |
| logistic | 0.0634 | 0.1416 | 0.1594 | 0.0615 | 0.1529 | 0.1619 | 3.22 |
| hist_gradient_boosting | 0.3027 | 0.5389 | 0.4609 | 0.3113 | 0.5549 | 0.4702 | 4.24 |

Modelo selecionado: **hist_gradient_boosting**.

Critério: maior Top-1 na validação, com MRR como desempate. Todos os
modelos ranqueiam somente lances legais; por isso a taxa de legalidade
esperada é 100%.

Depois da seleção, o vencedor foi retreinado com treino + validação.
No teste isolado, o artefato final obteve Top-1 de **0.3161**, Top-3 de **0.5601** e MRR de **0.4746**.

Os recortes completos por cor, fase, rating e ECO estão em
[`latest_metrics.json`](latest_metrics.json).
