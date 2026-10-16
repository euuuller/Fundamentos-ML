# Status — Exercício 02: Previsão de Sobrevivência no Titanic

Última atualização: 2026-10-12

## Andamento

| Etapa | Descrição | Guia | Situação |
|---|---|---|---|
| 0 | Análise da referência, especificação e planejamento | [`00_Plano.md`](00_Plano.md) | ✅ Concluída |
| 1 | Importações e obtenção dos dados | [`01_Importacoes_e_Dados.md`](01_Importacoes_e_Dados.md) | ✅ Concluída |
| 2 | Inspeção inicial, definição do alvo e EDA simples | [`02_Inspecao_e_Analise_Exploratoria.md`](02_Inspecao_e_Analise_Exploratoria.md) | ✅ Concluída |
| 3 | Separação dos dados de treino e teste | [`03_Separacao_Treino_Teste.md`](03_Separacao_Treino_Teste.md) | ✅ Concluída |
| 4 | Pré-processamento com `ColumnTransformer` | [`04_Preprocessamento.md`](04_Preprocessamento.md) | ✅ Concluída |
| 5 | Preparação e treinamento dos três modelos | [`05_Treinamento_dos_Modelos.md`](05_Treinamento_dos_Modelos.md) | ✅ Concluída |
| 6 | Validação cruzada e comparação das métricas (com AUC) | [`06_Validacao_Cruzada.md`](06_Validacao_Cruzada.md) | ✅ Concluída |
| 7 | *Trade-off*, curva ROC/AUC e limiar F1 | [`07_Tradeoff_ROC_Limiar.md`](07_Tradeoff_ROC_Limiar.md) | ✅ Concluída |
| 8 | Matrizes de confusão no treino (fora da amostra) | [`08_Matrizes_Confusao_Treino.md`](08_Matrizes_Confusao_Treino.md) | ✅ Concluída |
| 9 | Avaliação final no teste | [`09_Avaliacao_Teste.md`](09_Avaliacao_Teste.md) | ✅ Concluída |
| 10 | Comparação final, conclusão e notebook de apresentação | [`10_Conclusao.md`](10_Conclusao.md) | ✅ Concluída |

**Exercício 02 concluído.**
- Notebook de estudo: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb` (154 células, 68 de código), reproduzido em kernel limpo com saídas idênticas.
- Notebook de apresentação: `Exercise/2-Exercise.ipynb` (114 células: 68 de código + 46 títulos), com saídas idênticas às do notebook de estudo.
- Guias: `Docs/Guia/2-Exercise/`. Especificação: `Docs/Prompt/2-Prompt.md`.

## Registro da Etapa 0

- Seção EXERCÍCIO 2 da referência lida: enunciado (métricas com ROC/AUC), dicionário de atributos e `load_titanic_data()`.
- Dados inspecionados **em memória** (nada gravado): `train.csv` 891 × 12, `test.csv` 418 × 11 **sem `Survived`**; alvo 549/342 (38,38% sobreviventes); ausentes `Age` 177, `Cabin` 687, `Embarked` 2.
- Decisões do usuário: os mesmos 3 modelos; limiar pelo máximo F1; atributos básicos (`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`).
- Criados: `Docs/Prompt/2-Prompt.md` (especificação), este plano e este status.
- Pasta `datasets/` criada (com `.gitkeep`) e `.gitignore` atualizado (`datasets/*`, exceto `.gitkeep`). Nada foi baixado ainda.
- `CLAUDE.md` e `README.md` atualizados para o Exercício 02.
- Nenhum notebook criado, nenhum treino executado, nenhum commit.

## Registro da Etapa 1

- Notebook `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb` criado: título, "1. Observar o Panorama Geral" (com dicionário de atributos) e "2. Obtenção dos Dados" (início). São 14 células, 5 de código.
- `load_titanic_data()` da referência mantida, com o caminho adaptado para `PASTA_DATASETS = Path("../datasets")`. Dados baixados e extraídos para `datasets/titanic/` (raiz); `Notebooks/datasets/` não foi criado. O `git status` mostra só `datasets/.gitkeep`.
- Resultados: `concat` → (1309, 12) com **418** `Survived` nulos (o `test.csv` não tem rótulo); após `dropna(subset=['Survived'])` → **(891, 12)**.
- `Survived` está como `float` (0.0/1.0) por causa do `concat`; a conferência e a conversão ficam para a Etapa 2.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis criadas para as próximas etapas: `data`, `PASTA_DATASETS`.

## Registro da Etapa 2

- Seções "2.1. Definição do Alvo" e "2.2. Descrição dos Dados" acrescentadas (17 células, 11 de código); o notebook tem agora 31 células.
- Alvo conferido: só 0.0/1.0, já na codificação do enunciado (**sem inversão**), convertido para `int`. Distribuição: 549 (61,62%) × 342 (38,38%).
- Ausentes: `Cabin` 687 (77,10%), `Age` 177 (19,87%), `Embarked` 2 (0,22%).
- Associações nos dados: sobrevivência de 74,20% (mulheres) × 18,89% (homens); 62,96% / 47,28% / 24,24% (1ª/2ª/3ª classe).
- Atributos definidos: `atributos = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`. Descartados `PassengerId`, `Name` (891 distintos), `Ticket` (681 distintos) e `Cabin` (77,10% ausente).
- Correções durante a etapa: a legenda das barras ("Survived" → "Taxa do grupo") e o histograma da idade (faixas diferentes por classe → mesmas faixas de 5 anos).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `data` (com `Survived` int), `nomes_classes`, `atributos`.

## Registro da Etapa 3

- Seção "2.3. Separação dos Dados de Treino e Teste" acrescentada (6 células, 3 de código); o notebook tem agora 37 células.
- `X = data[atributos]` (DataFrame 891 × 7) e `y = data['Survived']`; `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.
- Treino (712, 7) com 273 sobreviventes; teste (179, 7) com 69 sobreviventes. Proporção de sobreviventes: 38,38% completo, 38,34% treino, 38,55% teste.
- Ausentes: `Age` 137 (treino) e 40 (teste); `Embarked` 2 (treino) e 0 (teste).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- `X_test` e `y_test` estão **reservados para a Etapa 9**.

## Registro da Etapa 4

- Seção "2.4. Pré-processamento com `ColumnTransformer`" acrescentada (9 células, 5 de código); o notebook tem agora 46 células.
- `preprocessamento = ColumnTransformer([('num', SimpleImputer(median) + StandardScaler, ['Pclass','Age','SibSp','Parch','Fare']), ('cat', SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore'), ['Sex','Embarked'])])`, **não ajustado**, para entrar nos Pipelines da Etapa 5.
- Inspeção com `prep_inspecao = clone(preprocessamento).fit(X_train)`: mediana de `Age` = 28,5 (treino) e de `Fare` = 14,454; moda de `Embarked` = 'S'; `Sex` → ['female', 'male']; `Embarked` → ['C', 'Q', 'S'].
- Treino transformado: (712, 7) → **(712, 10)**, `ndarray`, 0 ausentes. Exemplo conferido manualmente (Age ausente → 28,5 → −0,081).
- `X_test` não foi transformado nem usado.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `preprocessamento`, `atributos_numericos`, `atributos_categoricos`.

## Registro da Etapa 5

- Seção "3. Treinamento dos Modelos de Classificação Binária" acrescentada (17 células, 8 de código); o notebook tem agora 63 células.
- `sgd_clf = make_pipeline(clone(preprocessamento), SGDClassifier(class_weight='balanced', random_state=42))`; `dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)`; `forest_clf = make_pipeline(clone(preprocessamento), RandomForestClassifier(class_weight='balanced', random_state=42))`. Todos treinados só com `X_train`/`y_train`.
- **Ajuste em relação ao plano:** `clone(preprocessamento)` em **ambos** os Pipelines (o plano previa o objeto original no SGD), para manter o `preprocessamento` como modelo-base não ajustado.
- Pesos `balanced`: Não Sobreviveu 0,811 e Sobreviveu 1,304.
- Exemplo (primeiro sobrevivente do treino: homem, 3ª classe, sem idade): SGD → Não Sobreviveu, Dummy → Não Sobreviveu, RF → Sobreviveu. Possível memorização da RF; é só uma verificação, não uma métrica.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `sgd_clf`, `dummy_clf`, `forest_clf`.

## Dúvida do usuário (antes da Etapa 6): remover ou imputar os ausentes?

Respondida no terminal. Imputar dentro do Pipeline foi mantido porque:
- remover as linhas reduziria os dados de 891 para 712 passageiros (179 a menos, 52 deles sobreviventes);
- os passageiros sem idade não são aleatórios: 76,84% são da 3ª classe (contra 49,72% entre os com idade) e têm taxa de sobrevivência de 29,38% (contra 40,62%);
- um passageiro novo sem idade não poderia ser classificado.

Remover linhas não causa vazamento (não aprende parâmetros), mas perde e distorce dados. Os 2 ausentes de `Embarked` poderiam ser removidos sem prejuízo; a imputação foi mantida por uniformidade.

## Registro da Etapa 6

- Seções "4. Avaliação de Performance" (definições, incluindo ROC/AUC) e "4.1 Validação Cruzada" acrescentadas (27 células, 7 de código); o notebook tem agora 90 células.
- `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; métricas accuracy, precision e f1 (com `zero_division=0`), recall e **roc_auc**.
- Resultados (média ± desvio, 5 dobras, limiar padrão):
  - Dummy: acurácia 61,66%; precisão, recall e F1 = 0; AUC 50%.
  - SGD: acurácia 71,90% ± 4,40%; precisão 62,78% ± 6,81%; recall 69,58% ± 10,40%; F1 65,33% ± 5,34%; AUC 78,75% ± 3,76%.
  - RF: acurácia 78,51% ± 1,45%; precisão 71,03% ± 3,16%; recall 74,72% ± 1,45%; F1 72,76% ± 0,98%; AUC 86,31% ± 0,75%.
- A RF superou o SGD em F1 e AUC nas 5 dobras (vantagem consistente). O SGD foi instável (F1 de 56,07% a 71,43%).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `cv`, `metricas`, `nomes_metricas`, `resultados_sgd`, `resultados_dummy`, `resultados_forest`, `df_comparacao`.

## Registro da Etapa 7

- Subseção "Trade-off da Precisão e Recall, Curvas ROC e Ajuste do Limiar" acrescentada à seção 4 (15 células, 6 de código); o notebook tem agora 105 células.
- Pontuações fora da amostra (mesmo `cv`): `decision_function` (SGD) e `predict_proba[:, 1]` (RF). AUC fora da amostra: RF 0,862 e SGD 0,787.
- Limiares pelo máximo F1, só com o treino: **`limiar_sgd = -0.8203`** (abaixo do padrão 0) e **`limiar_forest = 0.5954`** (acima do padrão 0,5). Sem empates.
- Treino (fora da amostra): SGD com FN 83 → 61 e FP 117 → 138 (F1 65,52% → 68,06%); RF com FP 84 → 61 e FN 69 → 76 (F1 72,73% → 74,20%; acurácia 80,76%).
- Verificação auxiliar: o F1 é quase plano perto do máximo; a RF tem 234 limiares porque 129 passageiros do treino têm atributos idênticos a outro (folhas impuras).
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `y_scores_sgd`, `y_scores_forest`, `limiar_sgd`, `limiar_forest`, `df_limiares`.

## Registro da Etapa 8

- Seção "4.2 Matriz de Confusão" acrescentada (16 células, 7 de código); o notebook tem agora 121 células.
- `plot_matriz_confusao` (a mesma do Exercício 01, com siglas TN/FP/FN/TP) com rótulos `Não Sobreviveu (0)` / `Sobreviveu (1)`; será reutilizada na Etapa 9.
- 5 matrizes separadas, do treino e fora da amostra (TN/FP/FN/TP; total de erros): Dummy 439/0/273/0 (273); SGD padrão 322/117/83/190 (200); SGD F1 301/138/61/212 (199); RF padrão 355/84/69/204 (153); RF F1 378/61/76/197 (**137**). São consistentes com as Etapas 6 e 7.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.
- Variáveis para as próximas etapas: `plot_matriz_confusao`, `y_train_pred_*` (dummy, sgd, sgd_f1, forest, forest_f1).

## Registro da Etapa 9

- Seção "5. Teste" acrescentada (26 células, 14 de código): 5.1 SGD, 5.2 Dummy, 5.3 RF, **5.4 Curvas ROC no Teste** (nova) e 5.5 Resumo. O notebook tem agora 147 células.
- O teste foi usado **pela primeira e única vez**; os modelos da Etapa 5 e os limiares da Etapa 7 foram aplicados sem alteração.
- `metricas_teste(y_pred, y_score)` com acurácia, precisão, recall, F1 e AUC. Há 5 matrizes separadas, com siglas.
- Resultados no teste (110/69):
  - Dummy: acurácia 61,45%, FN 69, AUC 50%.
  - SGD padrão: acurácia 79,33%, precisão 78,57%, recall 63,77%, F1 70,40%, AUC 83,70% (FN 25, FP 12).
  - SGD F1: acurácia 75,98%, precisão 67,57%, recall 72,46%, F1 69,93% (FN 19, FP 24).
  - RF padrão: acurácia 81,01%, precisão 76,92%, recall 72,46%, F1 74,63%, AUC 83,04% (FN 19, FP 15).
  - **RF F1: acurácia 82,12%, precisão 80,33%, recall 71,01%, F1 75,38%** (FN 20, FP 12; 32 erros).
- Ajuste F1: ajudou a RF (−3 FP, +1 FN) e não ajudou o SGD (+6 TP, +12 FP, F1 levemente menor).
- AUC no teste praticamente empatada (SGD 83,70% × RF 83,04%), ao contrário da validação cruzada (RF 86,31% × SGD 78,75%). Com o teste pequeno e o SGD instável, a diferença não é conclusiva.
- Notebook executado em kernel limpo, sem avisos nas células, e validado com `nbformat`.

## Registro da Etapa 10

- Seção "6. Comparação Final e Conclusão" acrescentada (7 células, 2 de código): tabela treino (fora da amostra) × teste com AUC e FN/FP, gráfico F1/AUC (eixo a partir de 50%, com aviso), 6.1 Interpretação e 6.2 Conclusão. O notebook tem agora 154 células.
- Nenhum modelo, parâmetro ou limiar foi alterado (a única pontuação nova é a do Dummy no treino, para a AUC dele).
- Conclusão: **Random Forest com limiar F1** foi a melhor no teste (F1 75,38%, acurácia 82,12%, precisão 80,33%, 32 erros) e a mais consistente na validação cruzada. O SGD foi instável (melhor no teste que na validação cruzada; AUC no teste 83,70% × RF 83,04%). O limiar F1 ajudou a RF no treino e no teste, e o SGD só no treino. Há aviso de que as relações são associações, não causas.
- **Reprodutibilidade:** o notebook foi reexecutado em kernel limpo. A primeira comparação indicou 3 células diferentes, mas era só uma linha em branco causada pela divisão variável do stdout em blocos. Com o script de comparação ajustado (juntando os blocos de stdout), a terceira execução deu **saídas idênticas** nas 68 células, sem erros nem avisos.
- **Notebook de apresentação gerado:** `Exercise/2-Exercise.ipynb` (68 células de código + 46 títulos, pela regra do `CLAUDE.md`). Executado em kernel limpo: 0 erros, 0 avisos, 20 figuras, e as saídas são **idênticas** às do notebook de estudo. `../datasets` funciona a partir de `Exercise/`.
- Nenhum commit (aguardando a criação do repositório no GitHub pelo usuário).

## Decisões em vigor

Ver [`00_Plano.md` § 1.5](00_Plano.md): caminho `../datasets`; só linhas rotuladas; `ColumnTransformer` no Pipeline; `StratifiedKFold(5, shuffle=True, random_state=42)`; limiar por máximo F1 com previsões fora da amostra; ROC/AUC; teste só na Etapa 9.

## Pendências

Nenhuma.

## Próxima etapa

Todas as etapas do Exercício 02 foram concluídas. Próximo passo do usuário: criar o repositório no GitHub e fazer o commit inicial.
