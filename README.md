# botChess

[![CI](https://github.com/jhvlima/botChess/actions/workflows/ci.yml/badge.svg)](https://github.com/jhvlima/botChess/actions/workflows/ci.yml)

O botChess é um projeto experimental para comparar modelos que ranqueiam os
lances que uma pessoa provavelmente faria em uma posição de xadrez. O bot
sempre escolhe entre lances legais e oferece uma interface web local com dois
modos:

- **Imitador:** segue diretamente o ranking produzido pelo modelo;
- **Coerente:** preserva o estilo do ranking, mas evita alguns erros táticos
  rasos, como mate imediato, afogamento com vantagem e perdas materiais graves.

O repositório não contém partidas pessoais, datasets derivados nem modelos
treinados com essas partidas. A demonstração usa somente preferências sintéticas
definidas no próprio código.

## Executar a demonstração

É necessário Python 3.10 ou mais recente.

```bash
git clone https://github.com/jhvlima/botChess.git
cd botChess
make install
make demo
```

`make demo` cria `artifacts/demo_model.joblib` localmente, inicia o servidor em
`127.0.0.1:8000` e abre o tabuleiro no navegador. O artefato sintético serve
somente para experimentar a interface: ele não representa um jogador real nem
uma força calibrada de xadrez.

Na página é possível escolher brancas ou pretas, selecionar **Coerente** ou
**Imitador**, realizar promoções, virar o tabuleiro e começar novas partidas.
Use `Ctrl+C` no terminal para encerrar.

Para iniciar sem abrir o navegador:

```bash
.venv/bin/botchess-web \
  --model artifacts/demo_model.joblib \
  --no-browser
```

## Usar um modelo próprio

Se você já possui um modelo botChess gerado localmente:

```bash
make play MODEL=/caminho/para/meu_modelo.joblib
```

Também é possível consultar uma posição:

```bash
.venv/bin/botchess-predict \
  --model /caminho/para/meu_modelo.joblib \
  --fen 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3' \
  --top-k 5
```

Ou integrar o modelo a uma GUI compatível com UCI:

```bash
.venv/bin/botchess-uci --model /caminho/para/meu_modelo.joblib
```

Arquivos Joblib usam serialização baseada em pickle. Carregue somente modelos
gerados por você ou obtidos de uma fonte em que confia.

## Treinar com dados autorizados

O pipeline recebe arquivos mensais no esquema JSON da Published Data API do
Chess.com. Antes de coletar, processar ou treinar com dados de qualquer serviço,
confirme que você possui os direitos e as autorizações necessários.

Os termos atuais do Chess.com exigem autorização prévia por escrito para certos
usos automatizados e para desenvolver modelos, engines ou datasets de IA a
partir do conteúdo do serviço. A API ser pública não substitui essa autorização.
Consulte a [política de dados do projeto](DATA_POLICY.md), o
[User Agreement do Chess.com](https://www.chess.com/legal/user-agreement) e a
[documentação da PubAPI](https://support.chess.com/en/articles/9650547-what-is-the-pubapi-and-how-do-i-use-it).

Com fontes que você esteja autorizado a usar, coloque os JSONs em `partidas/` e
execute:

```bash
make pipeline PLAYER=MeuUsuario
```

As etapas também podem ser executadas separadamente:

```bash
make prepare PLAYER=MeuUsuario
make validate PLAYER=MeuUsuario
make train
make test
```

O fluxo oficial é:

```text
JSONs autorizados em partidas/
              │
              ▼
      preparação cronológica
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
    treino validação  teste
       │      │        │
       └──────┼────────┘
              ▼
 frequência / regressão logística / gradient boosting
              │
              ▼
      artifacts/final_model.joblib
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
      web   predição   UCI
```

O split é cronológico por partida. Posições repetidas em um corte anterior são
descartadas dos cortes posteriores, e o conjunto de teste fica isolado durante
a seleção do modelo.

O coletor opcional `botchess-download` permanece disponível para contextos em
que exista autorização escrita aplicável. Ele não é executado pelo Makefile e
exige uma confirmação explícita:

```bash
.venv/bin/botchess-download \
  --username MeuUsuario \
  --output-dir partidas \
  --confirm-authorization
```

## Soluções implementadas

As abordagens compartilham preparação, features e métricas, mas mantêm a lógica
de modelagem separada em `src/botchess/solutions/`:

- [`frequency`](solutions/frequency/README.md): baseline por frequência global;
- [`logistic`](solutions/logistic/README.md): classificador linear de candidatos;
- [`hist_gradient_boosting`](solutions/hist_gradient_boosting/README.md): árvores
  de gradient boosting;
- [`maia`](solutions/maia/README.md): alternativa pesquisada, não integrada ao
  pipeline atual.

Regressão linear não é usada porque o alvo é uma escolha discreta entre lances
legais. A regressão logística é a comparação linear apropriada para esse tipo de
classificação.

## Experimento histórico

Um experimento local anterior, com 976 partidas rápidas e 33.131 exemplos,
selecionou `hist_gradient_boosting`. Depois do retreino com treino e validação,
o modelo obteve no teste histórico:

| Métrica | Resultado |
|---|---:|
| Top-1 | 31,61% |
| Top-3 | 56,01% |
| Top-5 | 66,10% |
| MRR | 0,4746 |

Esses números documentam o estudo, mas o dataset e o artefato pessoal não são
distribuídos neste repositório. Portanto, um clone novo não reproduz sozinho
esse resultado. Os relatórios agregados estão em
[`reports/`](reports/model-comparison.md).

No recorte histórico de 923 posições de final, o modo Coerente melhorou
proteções táticas específicas sem usar Stockfish:

| Métrica de final | Imitador | Coerente |
|---|---:|---:|
| Coincidência Top-1 com o lance histórico | 33,04% | 39,76% |
| Mate em um escolhido quando disponível | 3/8 | 8/8 |
| Mate em um entregue apesar de alternativa segura | 2 | 0 |
| Afogamentos escolhidos | 2 | 0 |

Essas métricas não equivalem a uma estimativa de rating ou força geral.

## Como funciona o modo Coerente

O modo analisa o lance do bot, uma resposta imediata do adversário e uma
recaptura direta quando houver. Ele prioriza mate em um, rejeita lances que
permitem mate imediato, evita afogamento com vantagem material e limita perdas
materiais rasas.

A tolerância material diminui conforme a partida avança: a proteção não é
aplicada na abertura, aceita até 300 centipawns no meio-jogo e até 150 no final.
Isso reduz erros simples, mas não substitui uma engine com busca profunda.

## Desenvolvimento e testes

```bash
make check
```

A suíte usa apenas fixtures sintéticas. Ela cobre preparação e validação do
dataset, features, seleção de soluções, legalidade das jogadas, protocolo UCI,
modos Imitador/Coerente e o fluxo HTTP da interface. `make check` também usa
Node.js para validar a sintaxe do JavaScript; `make test` executa somente os
testes Python.

O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) roda essas
verificações no GitHub com Python 3.10 e 3.12. Orientações para mudanças estão
em [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Organização

```text
src/botchess/
├── solutions/                 modelos e contrato compartilhado
├── data.py                    preparação do dataset
├── validation.py              auditoria dos splits
├── training.py                comparação e seleção
├── demo.py                    modelo sintético da demonstração
├── predict.py                 CLI de inferência
├── uci.py                     adaptador UCI
├── play_modes.py              modos Imitador e Coerente
├── web.py                     servidor Flask local
├── web_game.py                regras e sessões
├── templates/                 página do tabuleiro
└── static/                    JavaScript, CSS e ícone

solutions/                     documentação por abordagem
reports/                       resultados históricos agregados
docs/research/                 pesquisa de seleção de modelos
tests/                         testes automatizados
artifacts/                     modelos locais ignorados pelo Git
partidas/ e data/              dados locais ignorados pelo Git
```

Checkouts de terceiros, arquivos de Colab, partidas, datasets derivados e
modelos treinados não são versionados.

## Limitações

- O projeto aprende padrões observados, inclusive erros; não substitui uma
  engine forte como Stockfish.
- O modo Imitador não olha respostas futuras, e o Coerente faz apenas uma busca
  tática rasa.
- As partidas da interface ficam somente na memória.
- A aplicação Flask é destinada ao uso local e não deve ser exposta diretamente
  à internet.
- A melhor abordagem pode mudar quando o conjunto de dados mudar.
- O projeto ainda não possui uma licença explícita; até que uma seja escolhida,
  não presuma permissão para redistribuir ou reutilizar o código.

## Referências

- [Política de dados deste repositório](DATA_POLICY.md)
- [Chess.com User Agreement](https://www.chess.com/legal/user-agreement)
- [Chess.com PubAPI](https://support.chess.com/en/articles/9650547-what-is-the-pubapi-and-how-do-i-use-it)
- [python-chess](https://python-chess.readthedocs.io/)
- [Maia Chess](https://maiachess.com/)
