# Artefatos locais

Nenhum modelo treinado é versionado. Esta pasta permanece no repositório apenas
para documentar os caminhos usados pelos comandos.

Para criar a demonstração sintética:

```bash
make demo-model
```

O comando grava `demo_model.joblib` a partir de preferências de lances definidas
em `src/botchess/demo.py`. Ele não usa partidas reais nem acessa a internet.

Treinamentos com fontes autorizadas gravam o vencedor em
`final_model.joblib` e os candidatos em `solutions/<nome>/model.joblib`. Todos
esses arquivos são ignorados pelo Git.

Arquivos Joblib utilizam serialização baseada em pickle e podem executar código
durante o carregamento. Carregue somente artefatos gerados por você ou obtidos
de uma fonte em que confia.
