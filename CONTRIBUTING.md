# Contribuindo

## Ambiente local

```bash
make install
make check
```

O projeto suporta Python 3.10 ou mais recente. Os testes usam somente fixtures
sintéticas e não precisam das partidas ou do modelo treinado. `make check`
também usa Node.js apenas para validar a sintaxe do JavaScript; sem Node, rode
`make test` para executar a suíte Python.

## Escopo dos arquivos

- código compartilhado: `src/botchess/`;
- abordagens de modelagem: `src/botchess/solutions/`;
- documentação de cada abordagem: `solutions/`;
- testes automatizados: `tests/`;
- métricas publicadas: `reports/`.

Não inclua no commit ambientes virtuais, downloads em `partidas/`, datasets em
`data/`, checkouts de terceiros ou qualquer modelo Joblib. A demonstração gera
um artefato sintético local com `make demo-model`.

Use somente dados para os quais você possua direitos e autorizações adequados.
Não envie partidas, identificadores de jogadores ou datasets derivados em uma
contribuição. Consulte [`DATA_POLICY.md`](DATA_POLICY.md) antes de alterar o
fluxo de coleta ou treinamento.

## Antes de abrir uma mudança

1. execute `make check`;
2. mantenha o split cronológico e o conjunto de teste isolado da seleção;
3. documente uma nova abordagem em `solutions/<nome>/README.md`;
4. trate métricas experimentais como resultados históricos, não como artefatos
   reproduzíveis sem a respectiva fonte autorizada;
5. não adicione binários de modelos, partidas ou dados derivados ao Git.
