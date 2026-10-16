# Plano — Exercício 03: Classificação Multiclasse de Vinhos (Wine)

Especificação oficial: [`Docs/Prompt/3-Prompt.md`](../../Prompt/3-Prompt.md).
Referência: [`Docs/reference/2-Arquivo.ipynb`](../../reference/2-Arquivo.ipynb) (MNIST multiclasse + EXERCÍCIO 3), que não deve ser alterada.
Notebook de estudo (a criar na Etapa 1): `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`.
Notebook de apresentação (a gerar na Etapa 8): `Exercise/3-Exercise.ipynb`.
Guias deste exercício: `Docs/Guia/3-Exercise/`. Andamento: [`Status.md`](Status.md).

---

## 1. Análise da referência

### 1.1 Fluxo da parte MNIST multiclasse

| Seção da referência | Conteúdo |
|---|---|
| 1. Observar o Panorama Geral | `assert` de versões, `plt.rc`; MNIST, "Classificação Multiclasse", SISO, lote; métricas listadas: acurácia, precisão, recall, matriz de confusão e "Curva ROC" (não calculada) |
| 2. Obtenção dos Dados | `fetch_openml('mnist_784')`; `plot_digit` e grade de dígitos; distribuição das 10 classes (de 11,25% a 9,02%); `train_test_split` estratificado 80/20 (56 000 / 14 000) |
| 3. Classificação Multiclasse | Modelos **nativos** (Regressão Logística, RF, Naive Bayes, Softmax) × **estritamente binários** (SGD, SVC); estratégias **OvR** (N classificadores) e **OvO** (N(N−1)/2) |
| 3.1 SGDClassifier | `StandardScaler` ajustado no treino inteiro; SGD treinado; `decision_function` com 10 pontuações de uma amostra; classe = `np.argmax` |
| 3.2 DummyClassifier | `most_frequent` e `uniform` (o `uniform` é treinado e nunca avaliado) |
| 3.3 RandomForest | `predict_proba` como proporção de votos das árvores para uma amostra |
| 3.4 SVC | Kernel RBF em dados padronizados; OvO com 45 binários; `decision_function` agregada de uma amostra |
| 4. Avaliação de Performance | Por modelo: `cross_validate(cv=3)` com `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`; 3 matrizes de confusão no treino (absoluta, relativa por linha e só erros normalizados); 4.5 gráfico-resumo |
| 5. Teste | Por modelo: `classification_report(digits=4)`, métricas macro e matriz relativa |
| EXERCÍCIO 3 | Enunciado do Wine e célula com `load_wine()` |

### 1.2 Resultados da referência (MNIST; só como contexto)

| Modelo | CV: acurácia / F1 macro | Teste: acurácia / F1 macro |
|---|---|---|
| Dummy (most_frequent) | 11,25% / 2,02% | 11,25% / 2,02% |
| SGD | 90,03% / 90,12% | 89,89% / 90,03% |
| Random Forest | 96,54% / 96,51% | 96,72% / 96,69% |
| SVC | 96,01% / 95,99% | 96,42% / 96,41% |

### 1.3 Os quatro modelos (parâmetros da referência)

| Modelo | Configuração | Estratégia multiclasse | Entrada |
|---|---|---|---|
| `SGDClassifier` | `class_weight='balanced'`, `max_iter=1000`, `tol=1e-3`, `random_state=42` | **OvR** (binário por natureza) | Padronizada (Pipeline) |
| `DummyClassifier` | `strategy="most_frequent"` | Linha de base (sem aprendizado) | Bruta |
| `RandomForestClassifier` | `n_estimators=100`, `random_state=42`, `n_jobs=-1` (sem `class_weight`, como na referência) | **Nativo** | Bruta |
| `SVC` | `kernel='rbf'`, `gamma='scale'`, `random_state=42` | **OvO** (binário por natureza) | Padronizada (Pipeline) |

### 1.4 Requisitos do Exercício 3

- Dataset: *Wine Recognition* (`sklearn.datasets.load_wine`), 178 amostras de 3 cultivares e 13 atributos químicos.
- Objetivo: determinar a cultivar (`class_0`, `class_1` ou `class_2`) a partir da análise físico-química.
- Supervisionado; classificação **multiclasse (3 classes)**; MISO (13 entradas, 1 saída); aprendizado em lote.
- Medidas de desempenho: **acurácia, precisão, recall, F1-score e matriz de confusão**.

### 1.5 Fatos verificados no ambiente (somente leitura)

- `load_wine()`: 178 × 13, **sem valores ausentes**; vem embutido no scikit-learn (sem download).
- Alvo: 0/1/2 = `class_0`/`class_1`/`class_2`, com **59 / 71 / 48** amostras (33,15% / 39,89% / 26,97%).
- **As amostras vêm ordenadas por classe** (todos os 0, depois os 1, depois os 2).
- Escalas muito diferentes: `proline` de 278 a 1680 (desvio 314,9) × `nonflavanoid_phenols` de 0,13 a 0,66 (desvio 0,124).
- Teste de 20%: 178 × 0,2 = 35,6, arredondado para cima = **36 amostras** (treino = **142**). No teste, cada amostra vale ≈2,8 p.p. de acurácia.
- Com N = 3 classes, **OvR treina 3 classificadores e OvO também** (3 × 2 / 2 = 3).
- Ambiente: Python 3.14, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4, matplotlib 3.11.2 (o mesmo dos exercícios anteriores).

### 1.6 Problemas da referência que não serão copiados

| # | Problema na referência | Consequência |
|---|---|---|
| a | `assert` de versões | Proibido pela especificação |
| b | `StandardScaler` ajustado no treino inteiro antes do `cross_validate` | Vazamento entre as dobras |
| c | Matrizes de confusão do treino feitas com `predict` no próprio treino | Matrizes otimistas (dentro da amostra) e incoerentes com a avaliação por validação cruzada |
| d | Matriz de erros do SVC com o título "Random Forest" | Erro de identificação |
| e | Dummy `uniform` treinado e não avaliado | Código sem uso |
| f | `len(... estimators_ ...) or 45` para mostrar o nº de modelos do SVC | Número imposto, não obtido do modelo |
| g | "Curva ROC" no panorama, sem cálculo | Promessa não cumprida |
| h | Célula do Wine com saídas (distribuição, min/média/máx) que o código visível não gera | Saídas não reproduzíveis |
| i | `np.argmax(scores)` usado como rótulo | Devolve a posição; só funciona porque as classes estão ordenadas |
| j | `plt.rc('font', size=9/10)` no meio do notebook | Muda o estilo de todos os gráficos seguintes |
| k | `cv=3`, sem `StratifiedKFold` explícito com `shuffle` | Dobras dependentes da ordem dos dados (crítico no Wine, ordenado por classe) |

### 1.7 Adaptações (e por quê)

| # | Referência | Exercício 03 | Justificativa |
|---|---|---|---|
| A1 | *Scaler* fora do Pipeline | `make_pipeline(StandardScaler(), SGDClassifier(...))` e `make_pipeline(StandardScaler(), SVC(...))` | Padronização ajustada só no treino de cada divisão (sem vazamento) |
| A2 | `cv=3` | Um único `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Alvo ordenado (exige embaralhar); 142 amostras de treino (5 dobras de ≈28); mesmas divisões para todos os modelos; padrão dos Ex. 01/02 |
| A3 | Matrizes do treino com `predict` no treino | `cross_val_predict` com o mesmo `cv` | Previsões fora da amostra |
| A4 | `np.argmax(scores)` | `modelo.classes_[np.argmax(scores)]` | Rótulo correto, independentemente da ordem das classes |
| A5 | Rótulos numéricos | Nomes legíveis (`class_0`, `class_1`, `class_2`) em tabelas, relatórios e matrizes | Clareza (exigência da especificação) |
| A6 | Avisos `UndefinedMetricWarning` no Dummy | `zero_division=0` na precisão macro e nos relatórios | O Dummy nunca prevê duas das três classes |
| A7 | `plt.rc` alterado várias vezes | Um único `plt.rc` no início | Gráficos consistentes |
| A8 | "Curva ROC" citada | Sem ROC e sem ajuste de limiar; decisão = classe de maior pontuação | O enunciado do Exercício 3 não pede ROC; em multiclasse não há um limiar único |
| A9 | Saídas do Wine sem código | Distribuição e escalas calculadas no notebook (Etapa 2) | Reprodutibilidade |
| A10 | Dummy `uniform` não avaliado | Não incluído | Decisão do usuário (os 4 modelos) |
| A11 | Nº de modelos do SVC imposto | Nº de binários obtido do objeto (`coef_.shape[0]` no SGD) e pela fórmula com `len(classes_)` no SVC | Sem atalhos artificiais |

### 1.8 Convenções globais

- Português em Markdown, comentários, títulos e mensagens.
- `random_state=42` em toda parte.
- Teste usado **somente na Etapa 7**.
- Avaliação no treino **sempre fora da amostra**, com o mesmo `cv`.
- Células curtas; funções só quando eliminarem repetição relevante (candidatas: a função de matrizes, com 12 figuras no treino e 4 no teste, e a de métricas do teste).
- O teste terá 36 amostras: diferenças de 1 ou 2 acertos entre modelos não serão tratadas como conclusivas.
- O que não for executado é registrado como **NÃO EXECUTADO**, com motivo e próximo passo.
- Sem commits.

---

## 2. Fluxo de trabalho por etapa

Para cada etapa solicitada:

1. Ler este plano, o [`Status.md`](Status.md) e as células já existentes no notebook.
2. Explicar brevemente o objetivo da etapa.
3. Acrescentar **somente** as células da etapa em `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, por script `nbformat` salvo em arquivo (não por heredoc; ver `CLAUDE.md`).
4. Executar o notebook (`uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`), conferir as saídas e **abrir os gráficos** antes de interpretá-los.
5. Escrever o guia `Docs/Guia/3-Exercise/XX_Nome_da_Etapa.md` com o modelo abaixo e **apenas valores realmente obtidos**.
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

**Notebook de apresentação** (Etapa 8): `Exercise/3-Exercise.ipynb` é gerado a partir do notebook de estudo, com todas as células de código e **somente os títulos**, pela regra do `CLAUDE.md`. Depois é executado, e as saídas são comparadas com as do notebook de estudo (juntando os blocos de stdout).

---

## 3. Etapas

### Etapa 1 — Importações e obtenção dos dados
*(Ref.: seção 1 + célula do EXERCÍCIO 3)* · Guia: `01_Importacoes_e_Dados.md`

- **Objetivo e conceitos:** enquadrar o problema e carregar os dados. Conceitos: classificação multiclasse (cada amostra pertence a exatamente uma de 3 classes), MISO, aprendizado em lote, métricas do enunciado; estrutura `Bunch`.
- **Implementação prevista:** título; Markdown "1. Observar o Panorama Geral" com o enunciado (com a nota de que ROC/limiar não fazem parte do exercício); `plt.rc` único; `wine = load_wine()`; `wine.keys()`; `X = pd.DataFrame(wine.data, columns=wine.feature_names)` (como no enunciado); `y = wine.target`; formatos; lista numerada dos 13 atributos.
- **Resultados esperados:** chaves do `Bunch`, formatos de `X` e `y`, lista dos atributos.
- **Critérios de conclusão:** notebook criado e executado sem erro; dados carregados; panorama escrito.

### Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples
*(Ref.: 2.1/2.2)* · Guia: `02_Inspecao_e_Analise_Exploratoria.md`

- **Objetivo e conceitos:** conhecer o alvo e os atributos. Conceitos: codificação de rótulos multiclasse, ordem das amostras (por que embaralhar), balanceamento entre classes, escala dos atributos, separação visual das classes.
- **Implementação prevista:**
  - `target_names` + contagem por código;
  - dicionário de nomes legíveis (`nomes_classes`);
  - evidência da **ordem por classe** (primeiros e últimos rótulos ou gráfico rótulo × índice);
  - contagem de nulos;
  - distribuição das classes (tabela de quantidade e % + barras em %, estilo da referência);
  - `describe()` com mín./média/máx. de todos os atributos, reproduzindo de forma correta o que a célula 74 da referência mostrava sem código;
  - 2 ou 3 atributos por classe (boxplot ou histograma), escolhidos e justificados na própria etapa com base nos dados.
- **Resultados esperados:** contagens por classe, gráfico de proporções, tabela de escalas e gráficos por classe.
- **Critérios de conclusão:** codificação confirmada; ordem por classe evidenciada; escalas e balanceamento descritos; nenhuma afirmação além do que os dados mostram.

### Etapa 3 — Separação dos dados de treino e teste
*(Ref.: 2.3)* · Guia: `03_Separacao_Treino_Teste.md`

- **Objetivo e conceitos:** reservar dados para a avaliação final. Conceitos: *hold-out*, estratificação em multiclasse, embaralhamento (crítico com dados ordenados), tamanho do teste num conjunto pequeno.
- **Implementação prevista:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` (o embaralhamento é o padrão da função); tabela de quantidade e proporção por classe (completo/treino/teste); formatos.
- **Resultados esperados:** formatos dos conjuntos e proporções por classe.
- **Critérios de conclusão:** proporções equivalentes nas três partes; teste reservado até a Etapa 7; o impacto do teste pequeno discutido.

### Etapa 4 — Estratégias multiclasse e treinamento dos quatro modelos
*(Ref.: seção 3 — 3.1 a 3.4)* · Guia: `04_Estrategias_e_Treinamento.md`

- **Objetivo e conceitos:** entender como cada modelo decide entre 3 classes e treiná-los. Conceitos: modelos nativos × binários; **OvR** (N classificadores, classe = maior pontuação) e **OvO** (N(N−1)/2 duelos, classe = mais votos/maior pontuação agregada); `decision_function` × `predict_proba`; Pipeline; `class_weight='balanced'`.
- **Implementação prevista:**
  - Markdown com as estratégias e as fórmulas (adaptado da referência).
  - **3.1 SGD:** `sgd_clf = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced', max_iter=1000, tol=1e-3, random_state=42))`; `fit`; pontuações (`decision_function`) de uma amostra do treino, classe por `sgd_clf.classes_[argmax]`; nº de binários OvR pelo modelo (`coef_.shape[0]`).
  - **3.2 Dummy:** `DummyClassifier(strategy="most_frequent")`; `fit`; previsão da mesma amostra.
  - **3.3 RF:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`; `fit`; `predict_proba` (votos das árvores) da amostra.
  - **3.4 SVC:** `svc_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', random_state=42))`; `fit`; pontuações agregadas da amostra; nº de pares OvO por N(N−1)/2 com `len(classes_)`.
  - Explicação de por que, com 3 classes, OvR e OvO têm 3 classificadores.
- **Resultados esperados:** modelos treinados; tabelas de pontuações/probabilidades da amostra; números de classificadores internos.
- **Critérios de conclusão:** 4 modelos treinados só com o treino; decisão multiclasse explicada com as pontuações reais; nenhuma métrica ainda.

### Etapa 5 — Avaliação com validação cruzada e comparação das métricas macro
*(Ref.: seção 4 — validação cruzada e 4.5 gráfico-resumo)* · Guia: `05_Validacao_Cruzada.md`

- **Objetivo e conceitos:** estimar o desempenho só com o treino. Conceitos: precisão, recall e F1 **por classe** (cada classe como positiva contra as demais); **média macro** (média simples entre classes), **ponderada** (pelo nº de amostras de cada classe) e **micro** (agregando todos os acertos; em multiclasse de rótulo único, micro = acurácia); validação cruzada estratificada.
- **Implementação prevista:** Markdown com as fórmulas; `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; `cross_validate` para os 4 modelos com `accuracy`, `precision_macro` (via `make_scorer(precision_score, average='macro', zero_division=0)`), `recall_macro` e `f1_macro`; média ± desvio; tabela de F1 macro por dobra; tabela-resumo + gráfico de barras (estilo 4.5 da referência).
- **Resultados esperados:** métricas dos 4 modelos. Por construção, o Dummy terá recall macro de 1/3 e acurácia próxima da proporção da classe majoritária no treino; os demais valores só serão conhecidos na execução.
- **Critérios de conclusão:** tabela e gráfico gerados; interpretação baseada nos números obtidos e na variação entre dobras.

### Etapa 6 — Matrizes de confusão no treino (fora da amostra) e análise por classe
*(Ref.: matrizes da seção 4, corrigidas)* · Guia: `06_Matrizes_Confusao_Treino.md`

- **Objetivo e conceitos:** ver **quais cultivares se confundem**. Conceitos: matriz de confusão N×N (diagonal = acertos), normalização por linha (= recall de cada classe), matriz de erros normalizados (`sample_weight` = erros, mostra para onde vão os erros de cada classe), previsões fora da amostra.
- **Implementação prevista:**
  - `cross_val_predict` (mesmo `cv`) para os 4 modelos;
  - função que desenha as **3 versões** (absoluta; `normalize="true"`; erros com `sample_weight=(pred != real)` e `normalize="true"`) com nomes das classes, eixos "Classe real"/"Classe prevista" e título com modelo e tipo;
  - `classification_report` fora da amostra por modelo (precisão, recall e F1 por classe);
  - tabela dos pares (classe real → classe prevista) mais confundidos.
- **Resultados esperados:** 12 matrizes (4 modelos × 3 tipos), relatórios por classe e a tabela de confusões.
- **Critérios de conclusão:** matrizes separadas e identificadas; confusões entre cultivares descritas com os valores obtidos; coerência com a Etapa 5. Se um modelo não errar nenhuma amostra de uma classe, a linha correspondente da matriz de erros fica zerada, e isso será explicado.

### Etapa 7 — Avaliação final no teste
*(Ref.: seção 5)* · Guia: `07_Avaliacao_Teste.md`

- **Objetivo e conceitos:** medir a generalização em dados nunca usados. Conceitos: relatório por classe (`support`), média macro × ponderada no relatório, efeito do teste pequeno.
- **Implementação prevista:** função `metricas_teste` (acurácia + precisão, recall e F1 macro, com `zero_division=0`); para cada modelo, `classification_report(y_test, y_pred, target_names=..., digits=4, zero_division=0)` e matriz **relativa** própria; tabela e gráfico-resumo.
- **Resultados esperados:** relatórios por classe, 4 matrizes relativas e o resumo do teste.
- **Critérios de conclusão:** teste usado uma única vez; nada ajustado depois.

### Etapa 8 — Comparação final, conclusão e notebook de apresentação
*(Fechamento)* · Guia: `08_Conclusao.md`

- **Objetivo e conceitos:** sintetizar, verificar a reprodutibilidade e preparar a apresentação. Conceitos: CV (5 avaliações) × teste (1 avaliação); incerteza com amostras pequenas; comparação entre estratégias (nativo, OvR, OvO).
- **Implementação prevista:** tabela CV (fora da amostra) × teste para os 4 modelos (acurácia + métricas macro); gráfico comparativo; 6.1 Interpretação e 6.2 Conclusão (limitações: 178 amostras, teste de 36, um único *split*, hiperparâmetros padrão; exercício acadêmico); **reexecução em kernel limpo** com comparação das saídas; **geração de `Exercise/3-Exercise.ipynb`** e conferência de saídas idênticas; roteiro da apresentação no guia; atualização de `README.md` e `CLAUDE.md`.
- **Resultados esperados:** tabela e conclusão baseadas nos resultados obtidos; dois notebooks reproduzíveis.
- **Critérios de conclusão:** notebooks executam do início ao fim sem erro; saídas idênticas entre estudo e apresentação; guias e status sincronizados.

---

## 4. Skills

Nenhuma skill local será criada. O fluxo por etapa, o modelo do guia e a regra do notebook de apresentação já estão documentados (aqui e no `CLAUDE.md`), e os scripts de apoio (geração da apresentação, comparação de saídas, verificação de links) são executados a partir do *scratchpad*. Uma skill repetiria essas instruções sem ganho concreto.
