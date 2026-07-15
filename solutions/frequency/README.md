# Frequência global

Baseline que conta quantas vezes cada movimento UCI foi jogado no treino. Em
cada posição, ele ordena somente os lances legais pelas contagens históricas e
aplica suavização de uma ocorrência aos movimentos nunca vistos.

- Código: `src/botchess/solutions/frequency/model.py`
- Custo: praticamente instantâneo; artefato com poucos KiB
- Resultado no teste comparativo: 13,21% Top-1 e MRR 0,2583
- Uso: referência mínima que qualquer modelo aprendido deve superar

Experimento isolado, sem substituir o modelo final principal:

```bash
.venv/bin/botchess-train \
  --models frequency \
  --data-dir data/processed \
  --artifacts-dir artifacts/experiments/frequency \
  --reports-dir reports/experiments/frequency
```
