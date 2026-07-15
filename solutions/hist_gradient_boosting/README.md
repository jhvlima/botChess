# HistGradientBoosting

Conjunto de árvores que aprende relações não lineares entre a posição e cada
lance candidato. Positivos recebem peso proporcional à quantidade de exemplos
negativos para compensar o desbalanceamento.

- Código: `src/botchess/solutions/hist_gradient_boosting/model.py`
- Resultado antes do retreino: 31,13% Top-1 e MRR 0,4702
- Resultado histórico após retreino: 31,61% Top-1, 56,01% Top-3 e MRR 0,4746
- Situação no experimento histórico: solução selecionada

O modelo histórico foi retreinado com os 28.689 exemplos de treino e validação.
O artefato não é distribuído; um novo treinamento autorizado grava o resultado
local em `artifacts/final_model.joblib`.

Experimento isolado:

```bash
.venv/bin/botchess-train \
  --models hist_gradient_boosting \
  --data-dir data/processed \
  --artifacts-dir artifacts/experiments/hist-gradient-boosting \
  --reports-dir reports/experiments/hist-gradient-boosting
```
