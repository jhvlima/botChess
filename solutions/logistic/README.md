# Regressão logística

Classificador linear binário treinado por SGD. Cada linha representa um par
posição-lance e informa se o candidato foi ou não a escolha real do jogador-alvo.
Na inferência, as probabilidades dos lances legais são normalizadas entre si.

- Código: `src/botchess/solutions/logistic/model.py`
- Representação: as 119 features compartilhadas do projeto
- Resultado no teste comparativo: 6,15% Top-1 e MRR 0,1619
- Situação: avaliada, mas inferior até mesmo ao baseline de frequência

Regressão linear comum não foi usada porque o alvo não é contínuo. A regressão
logística é a formulação linear apropriada para a decisão binária de cada
candidato.

Experimento isolado:

```bash
.venv/bin/botchess-train \
  --models logistic \
  --data-dir data/processed \
  --artifacts-dir artifacts/experiments/logistic \
  --reports-dir reports/experiments/logistic
```
