# Plano — Exercício 02: Previsão de Sobrevivência no Titanic

Especificação oficial: [`Docs/Prompt/2-Prompt.md`](../../Prompt/2-Prompt.md).
Referência: [`Docs/reference/1-Arquivo.ipynb`](../../reference/1-Arquivo.ipynb) (seção EXERCÍCIO 2), que não deve ser alterada.
Notebook de estudo (a criar na Etapa 1): `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`.
Notebook de apresentação (a gerar na Etapa 10): `Exercise/2-Exercise.ipynb`.
Dados: pasta `datasets/` da raiz (ignorada pelo git, exceto `.gitkeep`).
Guias deste exercício: `Docs/Guia/2-Exercise/`. Andamento: [`Status.md`](Status.md).

---

## 1. Análise da referência

### 1.1 O que a referência traz para o Exercício 2

| Célula | Conteúdo |
|---|---|
| Enunciado | Dataset Titanic; objetivo: prever se o passageiro sobreviveu; supervisionado; classificação binária (**0 = Não Sobreviveu, 1 = Sobreviveu**); MISO; aprendizado em lote; métricas: **acurácia, precisão, recall, F1, matriz de confusão, curva ROC e AUC** |
| Dicionário | `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked` (com significados) |
| `load_titanic_data()` | Baixa `https://github.com/ageron/data/raw/main/titanic.tgz` para `datasets/`, extrai e **concatena** `train.csv` + `test.csv` |
| Comentário final | "O objetivo é treinar um classificador que possa prever a coluna Survived com base nas outras colunas." |

O enunciado do Exercício 2 **não especifica os modelos**. Por decisão do usuário, serão usados **os mesmos três** da referência e do Exercício 01. O fluxo (Panorama → Dados → Treino → Avaliação → Teste) é o mesmo da seção MNIST da referência, já adaptado no Exercício 01.

### 1.2 Os três modelos

| Modelo | Configuração | Entrada |
|---|---|---|
| `SGDClassifier` | `class_weight='balanced'`, `random_state=42` | Pré-processada (imputação + one-hot + **padronização**) |
| `DummyClassifier` | `strategy="most_frequent"`, `random_state=42` | Qualquer (ignora os atributos) |
| `RandomForestClassifier` | `class_weight='balanced'`, `random_state=42` | Pré-processada (a padronização não a afeta) |

### 1.3 Fatos verificados (dados lidos em memória durante o planejamento; nada foi gravado)

- `titanic.tgz` (≈32 KB) contém `titanic/train.csv` (**891 × 12**) e `titanic/test.csv` (**418 × 11**).
- **O `test.csv` não tem `Survived`**: é o conjunto de submissão do Kaggle. O `concat` da referência gera 1309 linhas, das quais **418 têm alvo nulo**. Só as **891 linhas rotuladas** servem para treinar e avaliar.
- Alvo: 549 (61,62%) não sobreviveram e 342 (38,38%) sobreviveram. A codificação já é 1 = Sobreviveu, mas será conferida no notebook.
- Ausentes nas 891 linhas: `Age` 177, `Cabin` 687 (77%), `Embarked` 2; os demais atributos não têm ausentes.
- Tipos: `Sex` e `Embarked` são texto; `Pclass`, `SibSp`, `Parch` inteiros; `Age`, `Fare` reais; `Name`, `Ticket`, `Cabin` texto livre.
- Distribuições: `Sex` (577 M / 314 F); `Embarked` (S 644, C 168, Q 77); `Pclass` 1/2/3 = 216/184/491.

### 1.4 Decisões do usuário

1. **Modelos:** os mesmos 3.
2. **Atributos:** `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`; descartar `PassengerId` (identificador), `Name` e `Ticket` (texto livre, praticamente únicos por passageiro) e `Cabin` (77% ausente).
3. **Limiar:** *trade-off* + curva ROC/AUC; limiar pelo **máximo F1**, só com o treino.

### 1.5 Adaptações em relação à referência (e ao Exercício 01)

| # | Referência | Exercício 02 | Justificativa |
|---|---|---|---|
| A1 | `Path("datasets/...")` | `Path("../datasets/...")` | O notebook roda em `Notebooks/`; o caminho relativo da referência criaria `Notebooks/datasets/`. A pasta oficial é `datasets/`, na raiz |
| A2 | `concat(train_df, test_df)` | A função é mantida, e depois **ficam só as linhas com `Survived`** | O `test.csv` não tem rótulo e não serve para treino nem avaliação. Mostrar o problema e filtrar é mais didático que esconder |
| A3 | Sem pré-processamento (MNIST é numérico e completo) | `ColumnTransformer` dentro do Pipeline: numéricos `Age, SibSp, Parch, Fare, Pclass` → `SimpleImputer(strategy='median')` + `StandardScaler`; categóricos `Sex, Embarked` → `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` | Há ausentes e atributos de texto. Ajustar dentro do Pipeline garante que medianas, modas e escalas sejam aprendidas só no treino de cada dobra (sem vazamento) |
| A4 | `cv=3` | Um único `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Treino pequeno (712 passageiros); mesma escolha do Exercício 01; mesmas divisões para todos os modelos |
| A5 | Limiar ilustrativo (3000) | Limiar pelo **máximo F1** nas previsões fora da amostra do treino | Diferentemente do câncer, não há erro claramente mais grave; o F1 pesa FN e FP igualmente. Sem meta arbitrária e sem usar o teste |
| A6 | Sem ROC | `roc_auc` na validação cruzada; curvas ROC no treino (fora da amostra) e no teste | Exigido pelo enunciado do Exercício 2 |
| A7 | `assert` de versões; `X_test_array` inexistente | Não incluídos / corrigidos | Mesmos problemas tratados no Exercício 01 |

Detalhes de projeto:
- `Pclass` é tratado como **numérico ordinal** (1ª < 2ª < 3ª). É mais simples que *one-hot* e preserva a ordem. A alternativa será citada no guia.
- O mesmo pré-processador é usado (clonado) no SGD e na RF. A padronização não muda as divisões das árvores, e manter um único pré-processador simplifica o código.
- O Dummy não precisa de pré-processamento.
- Rótulos nos gráficos e matrizes: `['Não Sobreviveu (0)', 'Sobreviveu (1)']`.
- A função `plot_matriz_confusao` (com siglas TN/FP/FN/TP) é reaproveitada do Exercício 01, com os rótulos do Titanic.

### 1.6 Convenções globais

- Português em Markdown, comentários, títulos e mensagens.
- `random_state=42` em toda parte.
- Teste usado **somente na Etapa 9**.
- Avaliação no treino **sempre fora da amostra** (mesmo `cv`).
- Células curtas; funções só quando eliminarem repetição relevante.
- Relações atributo × sobrevivência descritas como **associações nos dados**, nunca como causas.
- Nada de commits.

---

## 2. Fluxo de trabalho por etapa

Para cada etapa solicitada:

1. Ler este plano, o [`Status.md`](Status.md) e as células já existentes no notebook.
2. Explicar brevemente o objetivo da etapa.
3. Acrescentar **somente** as células da etapa em `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, por script `nbformat` salvo em arquivo (não por heredoc; ver `CLAUDE.md`).
4. Executar o notebook (`uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`), conferir as saídas e **abrir os gráficos** antes de interpretá-los.
5. Escrever o guia `Docs/Guia/2-Exercise/XX_Nome_da_Etapa.md` com o modelo abaixo e **apenas valores realmente obtidos**.
6. Atualizar o `Status.md` e parar.

### Modelo do guia de cada etapa

1. **Objetivo** e relação com o exercício
2. **Código implementado**, por blocos
3. **Funções, parâmetros e variáveis** relevantes
4. **Conceitos e fórmulas**
5. **Justificativa das escolhas**
6. **Interpretação dos resultados obtidos**
7. **Erros comuns e limitações**
8. **Perguntas do professor**, com respostas curtas
9. **Como apresentar oralmente** a etapa
10. **Documentação oficial**: tabelas por biblioteca (Item | Para conferir | Link), versão *stable*, versões instaladas informadas, **cada link conferido** (inclusive âncoras `#...`)

**Notebook de apresentação** (Etapa 10): `Exercise/2-Exercise.ipynb` é gerado a partir do notebook de estudo, com todas as células de código e **somente os títulos**, pela mesma regra do Exercício 01 descrita no `CLAUDE.md`. Depois é executado, e as saídas são comparadas com as do notebook de estudo.

---

## 3. Etapas

### Etapa 1 — Importações e obtenção dos dados
*(Ref.: Panorama Geral + `load_titanic_data`)* · Guia: `01_Importacoes_e_Dados.md`

- **Objetivo e conceitos:** enquadrar o problema e obter os dados. Conceitos: supervisionado, classificação binária, MISO, lote, métricas (incluindo ROC/AUC); download e extração de arquivo `.tgz`; caminhos relativos.
- **Implementação prevista:** título; Markdown "Panorama Geral" (do enunciado) e dicionário de atributos; `plt.rc` como na referência; `load_titanic_data()` com `Path("../datasets")` (adaptação A1); `data = load_titanic_data()`; `shape` do resultado e contagem de `Survived` nulo; filtragem das linhas rotuladas (adaptação A2) e novo `shape`; `head()`.
- **Resultados esperados:** arquivos em `datasets/titanic/`; formato antes e depois da filtragem; primeiras linhas.
- **Critérios de conclusão:** dados baixados na pasta da raiz (ignorada pelo git); apenas linhas rotuladas em uso; motivo da filtragem explicado.

### Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples
*(Ref.: 2.1/2.2 do MNIST, adaptadas a dados tabulares)* · Guia: `02_Inspecao_e_Analise_Exploratoria.md`

- **Objetivo e conceitos:** conhecer tipos, ausentes e distribuições; confirmar Sobreviveu = 1; escolher os atributos. Conceitos: tipos de dados (numérico × categórico), valores ausentes, taxa por grupo, associação ≠ causalidade.
- **Implementação prevista:** `info()`; tabela de ausentes (quantidade e %); valores únicos de `Survived` e conversão para `int`; distribuição do alvo (tabela + barras em %); taxa de sobrevivência por `Sex` e por `Pclass` (barras); histograma de `Age` por classe; `describe()` dos numéricos; definição de `atributos` e justificativa dos descartes.
- **Resultados esperados:** tabelas e gráficos de distribuição; ausentes quantificados; lista final de atributos.
- **Critérios de conclusão:** codificação do alvo comprovada no notebook; atributos e descartes justificados; nenhuma afirmação causal.

### Etapa 3 — Separação dos dados de treino e teste
*(Ref.: 2.3)* · Guia: `03_Separacao_Treino_Teste.md`

- **Objetivo e conceitos:** reservar dados para a avaliação final. Conceitos: *hold-out*, estratificação, reprodutibilidade.
- **Implementação prevista:** `X = data[atributos]`, `y = data['Survived']`; `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`; tabela de quantidades e proporções (completo/treino/teste).
- **Resultados esperados:** formatos e proporções das classes nos três conjuntos.
- **Critérios de conclusão:** proporções equivalentes; teste reservado até a Etapa 9.

### Etapa 4 — Pré-processamento com `ColumnTransformer`
*(Nova; não existe na referência)* · Guia: `04_Preprocessamento.md`

- **Objetivo e conceitos:** transformar dados tabulares com ausentes e texto em uma matriz numérica, sem vazamento. Conceitos: imputação (mediana/moda), *one-hot encoding*, padronização, `ColumnTransformer`, `Pipeline`, ajuste só no treino.
- **Implementação prevista:** listas `atributos_numericos` e `atributos_categoricos`; `preprocessamento = ColumnTransformer([...])` (adaptação A3); para **inspeção**, ajustar um clone no `X_train`: medianas e modas aprendidas, `get_feature_names_out()`, formato da matriz transformada, verificação de que não restam nulos, primeiras linhas transformadas.
- **Resultados esperados:** valores aprendidos, nomes das colunas geradas e formato final.
- **Critérios de conclusão:** transformação entendida e justificada; ajuste feito só no treino.

### Etapa 5 — Preparação e treinamento dos três modelos
*(Ref.: seção 3)* · Guia: `05_Treinamento_dos_Modelos.md`

- **Objetivo e conceitos:** construir e treinar os 3 modelos. Conceitos: Pipeline completo (pré-processamento + modelo), `class_weight='balanced'`, *baseline*.
- **Implementação prevista:** pesos `balanced` (`compute_class_weight`); `sgd_clf = make_pipeline(clone(preprocessamento), SGDClassifier(...))` (*executado com `clone` também no SGD; ver Status*); `dummy_clf = DummyClassifier(...)`; `forest_clf = make_pipeline(clone(preprocessamento), RandomForestClassifier(...))`; `fit(X_train, y_train)`; previsão de um passageiro de exemplo (real × previsto nos 3 modelos).
- **Resultados esperados:** representações dos modelos, pesos das classes e previsões do exemplo.
- **Critérios de conclusão:** 3 modelos treinados só com o treino; escolhas justificadas; o exemplo é declarado como verificação, não como métrica.

### Etapa 6 — Avaliação com validação cruzada e comparação das métricas
*(Ref.: seção 4 e 4.1)* · Guia: `06_Validacao_Cruzada.md`

- **Objetivo e conceitos:** estimar o desempenho só com o treino. Conceitos: TP/TN/FP/FN no contexto do Titanic; fórmulas de acurácia, precisão, recall, F1; **curva ROC e AUC** (TPR × FPR; AUC como probabilidade de ordenar corretamente um par positivo-negativo); validação cruzada estratificada.
- **Implementação prevista:** Markdown com definições e fórmulas; `cv` (adaptação A4); `cross_validate` com `accuracy`, `precision` e `f1` (com `zero_division=0`), `recall` e `roc_auc`; média ± desvio; tabela por dobra; gráfico comparativo (estilo 4.1).
- **Resultados esperados:** métricas dos 3 modelos. Por definição, o Dummy terá recall 0 e AUC 0,5; os demais valores só serão conhecidos na execução.
- **Critérios de conclusão:** tabela e gráfico gerados; interpretação baseada nos números obtidos, incluindo a variação entre dobras.

### Etapa 7 — *Trade-off* precisão × recall, curva ROC/AUC e ajuste do limiar (F1)
*(Ref.: "Trade-off da Precisão e Recall")* · Guia: `07_Tradeoff_ROC_Limiar.md`

- **Objetivo e conceitos:** mostrar como o limiar troca FN por FP; comparar os modelos independentemente do limiar (ROC/AUC); escolher um limiar equilibrado. Conceitos: *scores*, curva P×R, curva ROC, AUC, F1 como critério.
- **Implementação prevista:** *scores* fora da amostra (`decision_function` no SGD; `predict_proba[:, 1]` na RF); `precision_recall_curve` e `roc_curve`/`roc_auc_score`; gráfico P×R (com zoom se necessário); **gráfico ROC** com SGD, RF e a diagonal do classificador aleatório, com AUC na legenda; precisão/recall × limiar; limiar = **máximo F1** (adaptação A5); tabela padrão × F1 (precisão, recall, F1, FN, FP).
- **Resultados esperados:** curvas, AUC fora da amostra, limiares escolhidos e variação de FN/FP.
- **Critérios de conclusão:** limiar escolhido só com o treino; diferença entre ROC (independe do limiar) e matriz/métricas (dependem do limiar) explicada.

### Etapa 8 — Matrizes de confusão da avaliação no treino (fora da amostra)
*(Ref.: 4.2)* · Guia: `08_Matrizes_Confusao_Treino.md`

- **Objetivo e conceitos:** ver os erros por tipo. Conceitos: leitura da matriz; FN = sobrevivente previsto como não sobrevivente; FP = o inverso.
- **Implementação prevista:** `plot_matriz_confusao` (do Exercício 01, com rótulos do Titanic e siglas TN/FP/FN/TP); 5 matrizes (Dummy, SGD padrão, SGD F1, RF padrão, RF F1) com `cross_val_predict` e os *scores* da Etapa 7; tabela-resumo TN/FP/FN/TP.
- **Resultados esperados:** 5 matrizes separadas e a tabela-resumo.
- **Critérios de conclusão:** matrizes identificadas e coerentes com as Etapas 6 e 7.

### Etapa 9 — Avaliação final no teste
*(Ref.: seção 5)* · Guia: `09_Avaliacao_Teste.md`

- **Objetivo e conceitos:** medir a generalização com dados nunca usados. Conceitos: `classification_report`, AUC no teste, aplicação de limiar fixado.
- **Implementação prevista:** função `metricas_teste` (acurácia, precisão, recall, F1, AUC); para cada variante, relatório, métricas e matriz própria; **curvas ROC no teste** (SGD × RF); tabela e gráfico-resumo (estilo 5.4).
- **Resultados esperados:** relatórios, 5 matrizes, curvas ROC e resumo do teste.
- **Critérios de conclusão:** teste usado uma única vez; nada ajustado depois.

### Etapa 10 — Comparação final, conclusão e notebook de apresentação
*(Fechamento)* · Guia: `10_Conclusao.md`

- **Objetivo e conceitos:** sintetizar, verificar a reprodutibilidade e preparar a apresentação. Conceitos: treino (fora da amostra) × teste; incerteza com amostras pequenas; associação × causalidade.
- **Implementação prevista:** tabela treino × teste (métricas + FN/FP + AUC); gráfico comparativo; 6.1 Interpretação e 6.2 Conclusão (limitações: 891 passageiros, um único *split*, sem engenharia de atributos, sem afirmações causais); **reexecução em kernel limpo** com comparação das saídas; **geração de `Exercise/2-Exercise.ipynb`** (código + títulos) e conferência de saídas idênticas; roteiro da apresentação no guia.
- **Resultados esperados:** tabela e conclusão baseadas nos resultados obtidos; dois notebooks reproduzíveis.
- **Critérios de conclusão:** notebooks executam do início ao fim sem erro; saídas idênticas entre estudo e apresentação; guias e status sincronizados.

---

## 4. Skills

Nenhuma skill local será criada. O fluxo por etapa, o modelo do guia e a regra do notebook de apresentação já estão documentados (aqui e no `CLAUDE.md`). Uma skill repetiria essas instruções sem ganho concreto.
