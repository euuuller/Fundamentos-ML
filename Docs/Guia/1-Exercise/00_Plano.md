# Plano — Exercício 01: Diagnóstico de Câncer e Otimização de Recall

Especificação oficial: [`Docs/Prompt/1-Prompt.md`](../../Prompt/1-Prompt.md).
Referência: [`Docs/reference/1-Arquivo.ipynb`](../../reference/1-Arquivo.ipynb), que não deve ser alterada.
Notebook de estudo (solução completa, com explicações): [`Notebooks/Exercicio_01_Cancer_Recall.ipynb`](../../../Notebooks/Exercicio_01_Cancer_Recall.ipynb).
Notebook de apresentação (células de código + títulos, gerado a partir do de estudo): [`Exercise/1-Exercise.ipynb`](../../../Exercise/1-Exercise.ipynb).
Guias deste exercício: `Docs/Guia/1-Exercise/`. Andamento: [`Status.md`](Status.md).

---

## 1. Análise da referência

### 1.1 O que a referência faz

O notebook de referência resolve uma classificação binária no **MNIST**: o dígito é 5 ou não é 5. O fluxo é:

| Seção da referência | Conteúdo |
|---|---|
| 1. Observar o Panorama Geral | Configuração do notebook (`plt.rc`) e enquadramento do problema: tipo de aprendizado, tarefa, SISO/MISO, aprendizado em lote, medidas de desempenho |
| 2. Obtenção dos Dados | Carregamento, inspeção (`X`, `y`, *shapes*), contagem de nulos, distribuição das classes (tabela + gráfico de barras em %), alvo binário, `train_test_split` estratificado |
| 3. Treinamento | Os três modelos, cada um com uma explicação teórica e um `predict` de exemplo |
| 4. Avaliação de Performance | Definições e fórmulas (acurácia, precisão, recall, F1); 4.1 `cross_validate` por modelo e gráfico comparativo; *trade-off* precisão × recall (`cross_val_predict` + `precision_recall_curve`); 4.2 matrizes de confusão com previsões fora da amostra |
| 5. Teste | Para cada modelo, `classification_report`, 4 métricas e matriz de confusão; 5.4 gráfico-resumo |
| Exercício 1 | Enunciado deste trabalho e célula com `load_breast_cancer()` |

### 1.2 Os três modelos (definidos na referência)

| Modelo | Configuração na referência | Dados de entrada |
|---|---|---|
| `SGDClassifier` | `class_weight='balanced'`, `random_state=42` | Padronizados (`StandardScaler`) |
| `DummyClassifier` | `strategy="most_frequent"`, `random_state=42` | Brutos (é só a *baseline*) |
| `RandomForestClassifier` | `class_weight='balanced'`, `random_state=42` | Brutos (árvores não dependem da escala) |

### 1.3 Requisitos do Exercício 1

- Classificar tumores de mama em **Maligno (1)** ou **Benigno (0)**.
- Fonte: `sklearn.datasets.load_breast_cancer()` (Wisconsin Diagnostic Breast Cancer). Vem embutido no scikit-learn e não precisa de download.
- Treinar e avaliar **os mesmos 3 modelos**.
- Métricas: **acurácia, precisão, recall e F1-score**, com foco em **recall** (título: "Otimização de Recall").

### 1.4 Fatos verificados no ambiente (somente leitura, sem treinamento)

- Ambiente `.venv` (uv): Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2, `jupyter`/`nbconvert` disponíveis. **Não é preciso instalar nada.**
- `load_breast_cancer()`: 569 amostras e 30 atributos numéricos.
- **Codificação original do alvo: `target_names = ['malignant', 'benign']`, ou seja, 0 = maligno e 1 = benigno** (212 malignos e 357 benignos). É o **inverso** do que o exercício pede, por isso o alvo será recodificado (Etapa 2).

### 1.5 Adaptações em relação à referência (e por quê)

| # | Referência | Nesta solução | Justificativa |
|---|---|---|---|
| A1 | Alvo já definido pelo problema | `y = 1 - cancer.target` | O dataset usa 0 = maligno. Para que "positivo" (recall, precisão, F1) se refira ao **maligno**, a classe precisa valer 1 |
| A2 | `scaler.fit_transform(X_train)` antes do `cross_validate` | `make_pipeline(StandardScaler(), SGDClassifier(...))` | Com o *Pipeline*, a padronização é reajustada só no treino de cada dobra e não há vazamento de informação da dobra de validação |
| A3 | `cv=3` (≈56 mil amostras de treino) | Um único `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` reutilizado em toda a validação | O treino aqui tem ≈455 amostras; 5 dobras estratificadas dão estimativas menos ruidosas, com ≈91 amostras por dobra e a proporção de malignos preservada. Usar o mesmo objeto deixa todas as comparações nas mesmas divisões |
| A4 | Limiar fixo `3000`, só ilustrativo | Limiar escolhido pelo máximo de **F2** nos *scores* fora da amostra do treino | Critério explícito que dá ao recall o dobro de peso da precisão, sem impor meta arbitrária de recall e sem tocar no teste |
| A5 | Célula 5.1 usa `X_test_array` (variável inexistente) | `X_test` direto; o Pipeline aplica o `transform` | Corrige um erro da referência |
| A6 | `assert` de versões | Não incluído | Proibido pela especificação; o ambiente já foi conferido |
| A7 | Rótulos `['Não é 5', 'É 5']` | `['Benigno (0)', 'Maligno (1)']`, eixos "Classe real" / "Classe prevista", título com o modelo | Matrizes separadas e identificadas, como pede a especificação |
| A8 | Avisos `UndefinedMetricWarning` no Dummy | `zero_division=0` nas métricas | O Dummy nunca prevê a classe positiva; o aviso é esperado e será explicado, não escondido sem comentário |

### 1.6 Convenções globais

- Comentários, Markdown, títulos de gráficos e mensagens em **português**.
- `random_state=42` em toda parte, para reprodutibilidade.
- O **teste** (`X_test`, `y_test`) só é usado na **Etapa 8**. Nenhuma escolha de modelo, parâmetro ou limiar o utiliza.
- Avaliação no treino **sempre fora da amostra** (`cross_validate` / `cross_val_predict` com o mesmo `cv`).
- Células curtas, sem classes; funções só quando eliminarem repetição relevante (candidata: plotar uma matriz de confusão, usada cerca de 10 vezes).
- Não há meta de desempenho nem promessa de resultado. O que não for executado é registrado como **NÃO EXECUTADO**, com o motivo e o próximo passo.

---

## 2. Fluxo de trabalho por etapa

Para cada etapa solicitada:

1. Ler este plano, o [`Status.md`](Status.md) e as células já existentes no notebook.
2. Explicar brevemente o objetivo da etapa.
3. Acrescentar **somente** as células da etapa em `Notebooks/Exercicio_01_Cancer_Recall.ipynb`.
4. Executar o notebook (`uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/Exercicio_01_Cancer_Recall.ipynb`) e conferir as saídas.
5. Escrever o guia `Docs/Guia/1-Exercise/XX_Nome_da_Etapa.md` usando o modelo abaixo e **apenas os valores realmente obtidos**.
6. Atualizar o `Status.md` e parar.

**Notebook de apresentação:** `Exercise/1-Exercise.ipynb` é **gerado** a partir do notebook de estudo e depois executado. Ele contém todas as células de código, na mesma ordem, e **somente os títulos** do Markdown: os cabeçalhos `#` e os subtítulos em negrito que antecedem código. Ficam de fora os textos explicativos, as "Observações" e os títulos sem código abaixo. Não deve ser editado à mão. Se o notebook de estudo mudar, o de apresentação precisa ser regerado e reexecutado, e as saídas dos dois devem ser conferidas.

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
10. **Documentação oficial**: tabelas por biblioteca (Item | Para conferir | Link) com a página oficial de cada função, classe e conceito usado **na etapa** (scikit-learn, NumPy, pandas, matplotlib, Python). Usar links da versão *stable*, informar as versões instaladas e **conferir se cada link responde** (inclusive âncoras `#...`) antes de publicar o guia.

---

## 3. Etapas

Cada etapa corresponde a uma ou mais seções do notebook, que seguem a numeração da referência (entre parênteses).

### Etapa 1 — Importações e obtenção dos dados
*(Ref. seção 1 "Panorama Geral" + início da seção 2)* · Guia: `01_Importacoes_e_Dados.md`

- **Objetivo e conceitos:** enquadrar o problema e carregar os dados. Conceitos: aprendizado supervisionado, classificação binária, MISO (30 entradas → 1 saída), aprendizado em lote, escolha das métricas; estrutura `Bunch` do scikit-learn (`data`, `target`, `feature_names`, `target_names`, `DESCR`).
- **Implementação prevista:**
  - Markdown de título e "Panorama Geral" (dataset, objetivo, tipo de aprendizado e de tarefa, entradas/saídas, formato, medidas de desempenho com ênfase em recall).
  - Importação do `matplotlib` e `plt.rc` como na referência (cada recurso é importado na etapa em que é usado; `numpy`/`pandas` entram na Etapa 2).
  - `cancer = load_breast_cancer()`; `X = cancer.data`; `y_original = cancer.target`.
  - Lista numerada dos 30 atributos (célula do enunciado) e `X.shape`, `y_original.shape`.
- **Resultados esperados:** lista dos 30 nomes de atributos e as dimensões de `X` e `y_original`. Nenhum gráfico.
- **Critérios de conclusão:** notebook criado, executa sem erro, dados carregados e Markdown de enquadramento presente.

### Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples
*(Ref. seções 2.1 e 2.2)* · Guia: `02_Inspecao_e_Analise_Exploratoria.md`

- **Objetivo e conceitos:** conhecer os dados e **garantir Maligno = 1**. Conceitos: codificação de rótulos, classe positiva, desbalanceamento moderado, escalas diferentes entre atributos.
- **Implementação prevista:**
  - `cancer.target_names` + contagem de `y_original`, mostrando que 0 = maligno no original.
  - `y = 1 - y_original` e uma tabela de conferência (original × novo × nome da classe).
  - `DataFrame` com `feature_names`, contagem de nulos e `describe()` de alguns atributos.
  - Tabela de quantidade e porcentagem por classe + gráfico de barras em % (estilo da seção 2.2 da referência), com rótulos "Benigno (0)" e "Maligno (1)".
  - Histograma de 1 ou 2 atributos separados por classe, evidenciando escalas diferentes (justifica a padronização do SGD).
- **Resultados esperados:** contagens por classe antes e depois da recodificação, gráfico de proporção, histogramas e tabela descritiva.
- **Critérios de conclusão:** a codificação Maligno = 1 / Benigno = 0 está **comprovada por saída do notebook**; o desbalanceamento foi quantificado e comentado.

### Etapa 3 — Separação dos dados de treino e teste
*(Ref. seção 2.3)* · Guia: `03_Separacao_Treino_Teste.md`

- **Objetivo e conceitos:** reservar dados nunca vistos para a avaliação final. Conceitos: generalização, *hold-out*, estratificação, reprodutibilidade (`random_state`).
- **Implementação prevista:**
  - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.
  - Proporção das classes em treino e teste e *shapes* (estilo da referência).
- **Resultados esperados:** *shapes* dos conjuntos e proporções de cada classe em treino e teste.
- **Critérios de conclusão:** as proporções são equivalentes entre os conjuntos; está justificado por que estratificar e por que o teste fica guardado até a Etapa 8.

### Etapa 4 — Preparação e treinamento dos três modelos
*(Ref. seção 3)* · Guia: `04_Treinamento_dos_Modelos.md`

- **Objetivo e conceitos:** construir e treinar os 3 modelos da referência. Conceitos: classificador linear e função de decisão (SGD), sensibilidade à escala, *Pipeline*, `class_weight='balanced'`, *baseline* (Dummy), *ensemble* de árvores (Random Forest).
- **Implementação prevista:**
  - `sgd_clf = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced', random_state=42))`.
  - `dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)`.
  - `forest_clf = RandomForestClassifier(class_weight='balanced', random_state=42)`.
  - `fit(X_train, y_train)` para cada um e `predict` de uma amostra de treino (como na referência), comparando com o rótulo real.
  - Markdown curto de cada modelo, adaptado da referência.
- **Resultados esperados:** representação dos estimadores ajustados e a previsão de exemplo de cada modelo. Nenhuma métrica ainda.
- **Critérios de conclusão:** 3 modelos treinados; justificados o *Pipeline* (adaptação A2) e o `class_weight`.

### Etapa 5 — Avaliação com validação cruzada e comparação das métricas
*(Ref. seção 4 — definições — e 4.1)* · Guia: `05_Validacao_Cruzada.md`

- **Objetivo e conceitos:** estimar o desempenho de cada modelo **só com o treino**. Conceitos: TP/TN/FP/FN, fórmulas de acurácia, precisão, recall e F1, validação cruzada k-fold estratificada, por que a acurácia engana quando há desbalanceamento, custo de um falso negativo em triagem de câncer.
- **Implementação prevista:**
  - Markdown com as fórmulas (adaptado da referência ao contexto clínico).
  - `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (adaptação A3).
  - `cross_validate` de cada modelo com acurácia, precisão, recall e F1 (precisão com `zero_division=0` via `make_scorer`, adaptação A8).
  - Tabela com média e desvio padrão por modelo + gráfico de barras comparativo (estilo 4.1).
- **Resultados esperados:** tabela de métricas médias (± desvio) dos 3 modelos e o gráfico comparativo. Pela definição do Dummy (prevê sempre a classe majoritária, Benigno), espera-se recall 0 para ele; os demais valores só serão conhecidos na execução.
- **Critérios de conclusão:** tabela e gráfico gerados; interpretação escrita **com base nos números obtidos**, destacando o recall.

### Etapa 6 — *Trade-off* entre precisão e recall e ajuste do limiar
*(Ref. "Trade-off da Precisão e Recall")* · Guia: `06_Tradeoff_Precisao_Recall.md`

- **Objetivo e conceitos:** mostrar que o limiar de decisão controla o equilíbrio entre FN e FP e escolher um limiar orientado a recall **sem usar o teste**. Conceitos: *score* de decisão × probabilidade, curva precisão-recall, limiar padrão (0 para `decision_function`, 0,5 para `predict_proba`), F-beta (F2).
- **Implementação prevista:**
  - *Scores* fora da amostra: `cross_val_predict(sgd_clf, ..., method="decision_function")` e `cross_val_predict(forest_clf, ..., method="predict_proba")[:, 1]`, com o mesmo `cv`. O Dummy não entra porque não tem *score* informativo.
  - `precision_recall_curve` para os dois modelos; curva P×R comparando SGD e RF (estilo da referência) e gráfico de precisão e recall × limiar.
  - Cálculo de F2 para cada limiar e escolha do máximo (adaptação A4); marcação do ponto do limiar padrão e do escolhido.
  - Tabela por modelo (limiar padrão × ajustado): precisão, recall, F1, nº de FP e nº de FN.
- **Resultados esperados:** curvas P×R, curvas × limiar, os limiares escolhidos e o quanto FN diminuem e FP aumentam (ou não) com o ajuste.
- **Critérios de conclusão:** limiar escolhido por critério explícito, apenas com previsões fora da amostra do treino; custo do aumento de recall (mais FP, menor precisão) explicado com os números obtidos.

### Etapa 7 — Matrizes de confusão da avaliação no treino (fora da amostra)
*(Ref. seção 4.2)* · Guia: `07_Matrizes_Confusao_Treino.md`

- **Objetivo e conceitos:** visualizar onde cada modelo erra no treino, sem otimismo de reavaliar dados já vistos. Conceitos: matriz de confusão, leitura de TN/FP/FN/TP, previsões fora da amostra.
- **Implementação prevista:**
  - `cross_val_predict` (mesmo `cv`) para os 3 modelos no limiar padrão.
  - Matrizes de SGD e RF no **limiar ajustado**, a partir dos *scores* da Etapa 6.
  - Uma figura por matriz, com `display_labels=['Benigno (0)', 'Maligno (1)']`, eixos "Classe real" / "Classe prevista" e título com o modelo e o limiar.
- **Resultados esperados:** 5 matrizes separadas (Dummy, SGD padrão, SGD ajustado, RF padrão, RF ajustado).
- **Critérios de conclusão:** matrizes identificadas; FN e FP de cada modelo comentados com os valores obtidos.

### Etapa 8 — Avaliação final no teste, com matrizes de confusão separadas
*(Ref. seções 5.1 a 5.4)* · Guia: `08_Avaliacao_Teste.md`

- **Objetivo e conceitos:** medir a generalização em dados nunca usados. Conceitos: estimativa final sem viés, `classification_report` (por classe, *macro* e *weighted avg*), aplicação de um limiar fixado previamente.
- **Implementação prevista:**
  - Para cada modelo (5.1 SGD, 5.2 Dummy, 5.3 RF): `predict(X_test)`, `classification_report`, as 4 métricas e uma matriz de confusão própria.
  - SGD e RF também com o **limiar da Etapa 6** (`score >= limiar`), com relatório e matriz próprios.
  - 5.4: tabela e gráfico-resumo de todas as variantes (estilo da referência).
- **Resultados esperados:** relatórios de classificação, matrizes separadas por modelo/limiar e o gráfico-resumo do teste.
- **Critérios de conclusão:** o teste foi usado uma única vez, e nenhum modelo, parâmetro ou limiar foi alterado depois de ver o teste.

### Etapa 9 — Comparação final, interpretação e conclusão
*(Fechamento, não presente na referência)* · Guia: `09_Conclusao.md`

- **Objetivo e conceitos:** sintetizar o que foi aprendido. Conceitos: comparação validação cruzada × teste, *trade-off* FN × FP em triagem, limitações do estudo.
- **Implementação prevista:**
  - Tabela final: métricas de validação cruzada (Etapa 5/6) lado a lado com as do teste (Etapa 8).
  - Markdown de conclusão: qual modelo/limiar teve melhor recall e a que custo em precisão e FP; papel do Dummy como *baseline*; limitações (amostra pequena, um único *split* de teste, dados de um único hospital/época). Deixar explícito que se trata de **exercício acadêmico, sem validade clínica**.
  - **Reexecução completa em kernel limpo** (`nbconvert --execute`) e conferência de que os números batem com os guias.
- **Resultados esperados:** tabela comparativa final e texto de conclusão baseado apenas nos resultados obtidos.
- **Critérios de conclusão:** notebook reproduzível do início ao fim sem erro; guias e status sincronizados com as saídas finais.
