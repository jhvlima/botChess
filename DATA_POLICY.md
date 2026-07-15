# Política de dados

Este repositório distribui código, testes com fixtures sintéticas, documentação
e métricas históricas agregadas. Ele não distribui:

- arquivos de partidas de jogadores;
- identificadores, URLs ou metadados de adversários;
- splits de treino, validação ou teste;
- modelos treinados com partidas pessoais;
- checkouts, pesos ou datasets de terceiros.

`partidas/`, `data/` e modelos em `artifacts/` são ignorados pelo Git. O modelo
criado por `make demo-model` usa somente uma pequena tabela sintética escrita no
código e também não é versionado.

## Fontes externas

Ser tecnicamente possível acessar um dado não significa que ele possa ser usado
para qualquer finalidade. Antes de coletar, armazenar, publicar ou treinar com
uma fonte externa, verifique:

1. os termos atuais do serviço;
2. a licença ou autorização aplicável aos dados;
3. direitos de privacidade e redistribuição de outras pessoas;
4. se a finalidade de treinamento de modelos está expressamente permitida.

Na revisão feita em 15 de julho de 2026, a seção 4.D do
[User Agreement do Chess.com](https://www.chess.com/legal/user-agreement)
exigia autorização prévia por escrito para certos usos automatizados e para
desenvolver ou melhorar modelos, engines e datasets de IA com conteúdo do
serviço. A [documentação da PubAPI](https://support.chess.com/en/articles/9650547-what-is-the-pubapi-and-how-do-i-use-it)
explica o acesso de leitura aos dados publicados, mas esse acesso não deve ser
interpretado como autorização adicional para treinamento.

O comando opcional `botchess-download` só deve ser usado quando o responsável
possuir autorização aplicável. Por isso ele exige `--confirm-authorization` e
não faz parte de `make pipeline`.

## Artefatos e segurança

Modelos Joblib usam o mecanismo de serialização do Python. Não carregue um
arquivo não confiável: ele pode executar código durante a desserialização.

Resultados em `reports/` são registros agregados de um experimento local
anterior. Eles não concedem direitos sobre a fonte usada e não tornam o estudo
reproduzível sem uma fonte autorizada equivalente.

Este documento registra decisões conservadoras de empacotamento do projeto e
não constitui aconselhamento jurídico.
