# Prompt CTRO — Exercício 03: Classificação Multiclasse de Vinhos (Wine)

## C — Contexto

Atue como especialista em Machine Learning, programação Python e ensino de Ciência de Dados.

Vou desenvolver o **Exercício 03 — Classificação Multiclasse de Vinhos**. O objetivo é determinar a qual das **3 cultivares** (`class_0`, `class_1` ou `class_2`) pertence uma garrafa de vinho, a partir de **13 atributos físico-químicos** (álcool, ácido málico, cinzas, magnésio, fenóis totais, flavonoides, intensidade de cor, prolina etc.) do *Wine Recognition Dataset* (`sklearn.datasets.load_wine`, 178 amostras).

Serão usados os **quatro modelos da seção multiclasse da atividade de referência**:

- `SGDClassifier`: modelo estritamente binário, que no multiclasse usa a estratégia **Um-contra-o-Resto (OvR)**, com dados padronizados;
- `DummyClassifier(strategy="most_frequent")`, como linha de base;
- `RandomForestClassifier`: modelo **nativamente multiclasse**;
- `SVC(kernel='rbf', gamma='scale')`: modelo estritamente binário, que no multiclasse usa a estratégia **Um-contra-Um (OvO)**, com dados padronizados.

Medidas de desempenho, conforme o enunciado: **acurácia, precisão, recall, F1-score e matriz de confusão**. Em multiclasse, precisão, recall e F1 são calculados **por classe** e resumidos pela **média macro**, como na referência.

O arquivo de referência está em:

`Docs/reference/2-Arquivo.ipynb`

Ele contém a resolução do **MNIST multiclasse** (modelos, estratégias OvR/OvO, validação cruzada com métricas macro, matrizes de confusão e teste) e, ao final, o enunciado do **EXERCÍCIO 3**.

O ambiente virtual com `uv` já está pronto, e os Exercícios 01 e 02 foram concluídos no mesmo repositório, seguindo o mesmo modelo de trabalho. Leia as instruções existentes no projeto (`CLAUDE.md`, `README.md`) e confira o ambiente antes de propor alterações.

Quero construir a resolução **por etapas**, compreender as decisões e conseguir explicar os códigos e resultados ao professor.

## T — Tarefas

### 1. Primeira execução: analisar e planejar

Nesta primeira interação:

1. Leia integralmente o notebook de referência, incluindo células Markdown, códigos, gráficos e resultados disponíveis, tanto da parte MNIST quanto do enunciado do Exercício 3.
2. Identifique os modelos, as estratégias multiclasse (nativo, OvR, OvO), a origem dos dados, o fluxo da atividade e os requisitos do exercício.
3. Identifique os **problemas da referência** que não devem ser copiados (ver "Rigor metodológico").
4. Inspecione o `CLAUDE.md`, o `README.md`, o `pyproject.toml` e a organização dos exercícios anteriores. Aproveite o ambiente existente.
5. Desenvolva um plano completo e numerado, acompanhando a estrutura da referência.
6. Salve o plano em `Docs/Guia/3-Exercise/00_Plano.md`, indicando, para cada etapa:
   - objetivo e conceitos envolvidos;
   - implementação prevista;
   - resultados e gráficos esperados, sem inventar valores;
   - critérios de conclusão.
7. Crie `Docs/Guia/3-Exercise/Status.md` com o andamento e a próxima etapa.
8. Avalie se alguma skill local seria realmente útil. Crie skills em `.claude/skills/` somente quando houver uma necessidade concreta, explicando a finalidade e evitando instruções duplicadas.

**Encerre após apresentar o plano. Não implemente a resolução nem execute treinamentos nesta primeira interação. Aguarde meu pedido para iniciar a Etapa 1.**

Se o arquivo de referência estiver ausente ou não permitir identificar um requisito essencial, informe a lacuna e solicite o material necessário.

### 2. Organizar a resolução em etapas

Ajuste a sequência abaixo ao notebook de referência:

1. Importações e obtenção dos dados.
2. Inspeção inicial, definição do alvo e análise exploratória simples (codificação 0/1/2 ↔ `class_0`/`class_1`/`class_2`, ordem das amostras, distribuição das classes, escalas dos atributos e atributos por classe).
3. Separação dos dados de treino e teste.
4. Estratégias multiclasse (modelos nativos, OvR e OvO) e treinamento dos quatro modelos, com a análise das pontuações de uma amostra e do número de classificadores internos de cada estratégia.
5. Avaliação com validação cruzada e comparação das métricas macro (com a explicação de média macro, ponderada e micro).
6. Matrizes de confusão da avaliação no treino, com previsões fora da amostra (absoluta, relativa e somente erros), e análise por classe (quais cultivares se confundem).
7. Avaliação final no teste, com relatório por classe, métricas macro e matrizes de confusão separadas.
8. Comparação final, interpretação dos resultados, conclusão e notebook de apresentação.

Preserve a ordem e o estilo da referência e dos exercícios anteriores sempre que forem metodologicamente adequados. Explique adaptações necessárias.

### 3. Executar uma etapa por solicitação

Quando eu solicitar uma etapa:

1. Leia o plano, o status e as células já desenvolvidas.
2. Explique brevemente o objetivo da etapa.
3. Implemente **somente a etapa solicitada**.
4. Acrescente as células ao mesmo arquivo:
   `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`.
5. Execute as células necessárias e confira os resultados, inclusive os gráficos. Não avance para etapas futuras.
6. Escreva o guia correspondente em:
   `Docs/Guia/3-Exercise/XX_Nome_da_Etapa.md`.
7. Atualize o status e pare, aguardando minha próxima solicitação.

Preserve o trabalho anterior. Se encontrar um erro que comprometa a etapa atual, corrija-o e registre o motivo.

### 4. Ensinar e documentar

Mantenha o notebook enxuto, com células curtas e explicações objetivas.

No guia de cada etapa, explique como um professor:

- o objetivo e a relação com o exercício;
- o código efetivamente implementado, por blocos;
- as funções, parâmetros e variáveis relevantes;
- os conceitos e as fórmulas necessárias;
- a justificativa das escolhas;
- a interpretação dos resultados realmente obtidos;
- erros comuns e limitações;
- perguntas que o professor poderia fazer, com respostas curtas;
- uma sugestão de como apresentar oralmente aquela etapa;
- a **documentação oficial** de cada função, classe e conceito usado na etapa, com links conferidos.

Explique detalhadamente o raciocínio, sem repetir informações desnecessárias. Mantenha o guia sincronizado com o notebook.

## R — Restrições

### Controle de execução

- É proibido resolver toda a atividade de uma vez.
- Planejar todas as etapas não autoriza executá-las.
- Execute apenas a etapa explicitamente solicitada.
- Não antecipe códigos, resultados ou conclusões de etapas futuras.
- Não altere os notebooks de referência nem os arquivos dos Exercícios 01 e 02.
- Não crie notebooks duplicados ou versões paralelas da solução.
- **Não faça commits** no repositório.

### Simplicidade

- Use código limpo, direto e legível, adequado ao nível acadêmico da referência.
- Evite classes, abstrações, funções auxiliares e configurações sem necessidade.
- Crie funções somente quando reduzirem repetição relevante ou organizarem uma operação clara.
- Não inclua verificações de versão com `assert`.
- Não reinstale dependências nem recrie o ambiente sem necessidade.
- Não adicione frameworks, hooks ou estruturas extras sem justificativa concreta.

### Rigor metodológico

- **Alvo:** confirme a codificação original (`target_names`) e use nomes legíveis das cultivares em tabelas, relatórios e matrizes.
- **Ordem das amostras:** o `load_wine()` entrega as amostras **ordenadas por classe**. Toda divisão deve embaralhar os dados (`train_test_split` e `StratifiedKFold(shuffle=True)`).
- **Divisão:** separe o teste com estratificação (`stratify=y`) e justifique a escolha, considerando o tamanho pequeno do conjunto.
- **Validação:** use um único `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` em todas as avaliações no treino.
- **Sem vazamento:** a padronização do SGD e do SVC deve ficar **dentro de um `Pipeline`** com `StandardScaler`, ajustada apenas nos dados de treino de cada divisão.
- **Teste:** reserve-o para a avaliação final. Não o utilize para escolher modelos, hiperparâmetros ou configurações.
- **Matrizes no treino:** use previsões **fora da amostra** (`cross_val_predict`), **nunca** `predict` no próprio conjunto de treino, como faz a referência.
- **Métricas multiclasse:** use precisão, recall e F1 com **média macro** (como a referência) e explique a diferença entre macro, ponderada (*weighted*) e micro. Use `zero_division=0` onde o Dummy produzir divisão por zero.
- **Decisão multiclasse:** explique que a classe prevista é a de **maior pontuação** (`argmax`). Ao mostrar a classe a partir das pontuações, use `modelo.classes_[argmax]`, e não a posição do `argmax` diretamente. Não faça ajuste de limiar nem curvas ROC: o enunciado do Exercício 3 não os pede.
- **Estratégias:** mostre o número de classificadores binários internos de cada estratégia a partir dos próprios objetos ou das fórmulas (N para OvR; N(N−1)/2 para OvO), sem atalhos artificiais. Com 3 classes, os dois números coincidem; explique isso.
- **Gráficos:** apresente matrizes de confusão separadas por modelo, com as classes e os eixos identificados. Não altere `plt.rc` no meio do notebook.
- **Erros da referência a não copiar:**
  - `assert` de versões;
  - `StandardScaler` ajustado antes da validação cruzada;
  - matrizes de confusão do treino calculadas dentro da amostra;
  - título trocado na matriz do SVC;
  - Dummy `uniform` treinado e não avaliado;
  - linha artificial para exibir o número de modelos do SVC;
  - "Curva ROC" listada e não calculada;
  - saídas da célula do Exercício 3 que o código visível não gera;
  - `np.argmax` usado como rótulo;
  - `plt.rc` alterado no meio do notebook.
- **Interpretação:** o conjunto é pequeno (178 amostras; o teste terá poucas dezenas). Interprete as diferenças entre modelos considerando a variação entre dobras e o número de amostras envolvido, sem prometer desempenho.
- Não invente métricas ou interpretações. Se não conseguir executar algo, registre **NÃO EXECUTADO**, o motivo e o próximo passo.
- Apresente a conclusão como resultado de um exercício acadêmico.
- Na etapa final:
  - execute o notebook completo em kernel limpo para verificar a reprodução dos resultados;
  - gere o notebook de apresentação `Exercise/3-Exercise.ipynb` (somente código e títulos, pela regra do `CLAUDE.md`), conferindo que as saídas são idênticas às do notebook de estudo.

## O — Saídas / Output

### Na primeira interação

Entregue:

1. Resumo da análise do arquivo de referência.
2. Plano completo em `Docs/Guia/3-Exercise/00_Plano.md`.
3. Controle de andamento em `Docs/Guia/3-Exercise/Status.md`.
4. Justificativa sobre a necessidade ou não de skills.
5. Indicação da primeira etapa a executar.

**Pare e aguarde meu pedido.**

### Em cada etapa posterior

Entregue:

1. Notebook atualizado apenas com o trabalho autorizado.
2. Guia didático da etapa em `Docs/Guia/3-Exercise/`.
3. Status atualizado.
4. Resumo breve no terminal contendo:
   - o que foi feito;
   - arquivos alterados;
   - verificações e resultados;
   - eventuais pendências;
   - próxima etapa proposta.

Ao concluir cada etapa, aguarde minha solicitação para continuar.
