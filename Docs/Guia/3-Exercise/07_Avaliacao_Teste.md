# Etapa 7 — Avaliação final no teste

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seção **5. Teste** (5.1 SGDClassifier, 5.2 DummyClassifier, 5.3 Random Forest, 5.4 SVC, 5.5 Resumo do Teste).
Correspondência na referência: seção 5 de `Docs/reference/2-Arquivo.ipynb` (`classification_report(digits=4)`, métricas macro e matriz relativa por modelo). Adaptações: nomes das classes no relatório e nas matrizes, `zero_division=0` também no relatório do Dummy, `X_test` bruto nos Pipelines (sem `scaler.transform` manual) e uma subseção de resumo (5.5), que a referência não tem para o teste.

---

## 1. Objetivo e relação com o exercício

Esta é a **avaliação final**: medir o desempenho dos 4 modelos em vinhos que **não participaram de nenhuma decisão**. O conjunto de teste (36 amostras: 12 / 14 / 10) é usado **aqui pela única vez**, com os modelos treinados na Etapa 4, sem retreino nem ajuste.

Para cada modelo: o relatório por classe, a acurácia e as métricas **macro** do enunciado e a matriz de confusão **relativa**. Depois, um resumo comparativo.

---

## 2. Código implementado, por blocos

### Bloco 1 — Introdução e função de métricas

```python
def metricas_teste(y_pred):
    resultado = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro'),
        'F1-Score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    ...
```

As mesmas quatro linhas da referência (acurácia, precisão, recall e F1 macro), transformadas em função porque se repetem nos 4 modelos. A função também **retorna** os valores para o resumo.

### Bloco 2 — Cada modelo (5.1 a 5.4)

```python
y_pred_sgd = modelos['SGD Classifier'].predict(X_test)
print(classification_report(y_test, y_pred_sgd, target_names=rotulos, digits=4, zero_division=0))
teste_sgd = metricas_teste(y_pred_sgd)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_sgd, display_labels=rotulos,
                                        normalize="true", values_format=".0%", cmap=plt.cm.Blues)
```

- `modelos[...]` é o dicionário da Etapa 6, com os mesmos objetos treinados na Etapa 4.
- O **Pipeline** do SGD e do SVC aplica ao `X_test` a padronização **aprendida no treino**. Na referência, era preciso chamar `scaler.transform(X_test)` à mão.
- O relatório usa `digits=4`, como na referência, `target_names` (nomes das cultivares) e `zero_division=0`, para que o Dummy não gere avisos.
- A matriz é a **relativa** (decisão do usuário para o teste, como na referência), com eixos e título identificados.

### Bloco 3 — Resumo (5.5)

Tabela com as 4 métricas e o **número de erros (de 36)** de cada modelo, e gráfico de barras no estilo da referência (`tab10`).

### Bloco 4 — Observações (Markdown)

**Correção durante a etapa:** o título da matriz do SGD saiu como "Sgd Classifier", por um `.title()` no script que gerou as células. Foi corrigido para "SGD Classifier" e o notebook foi reexecutado.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `classification_report(..., target_names=rotulos, digits=4, zero_division=0)` | Precisão, recall, F1 e *support* por classe; acurácia; médias macro e ponderada |
| `metricas_teste(y_pred)` | Calcula, imprime e retorna acurácia + métricas macro |
| `modelos[nome].predict(X_test)` | Previsão no teste (com padronização do treino nos Pipelines) |
| `ConfusionMatrixDisplay.from_predictions(..., normalize="true")` | Matriz relativa (diagonal = recall por classe) |
| `y_pred_sgd`, `y_pred_dummy`, `y_pred_forest`, `y_pred_svc` | Previsões no teste |
| `teste_*` | Métricas de cada modelo no teste |
| `df_teste` | Tabela-resumo com o número de erros |

---

## 4. Conceitos e fórmulas

**Leitura do `classification_report` multiclasse:**
- uma linha por cultivar, com precisão, recall, F1 e *support* (amostras reais da classe no teste: 12, 14 e 10);
- `accuracy`: acertos / 36;
- `macro avg`: média simples das 3 linhas (métrica do exercício);
- `weighted avg`: média ponderada pelo *support*.

**Granularidade do teste:**

$$\Delta Acc = \frac{1}{36} \approx 2{,}8 \text{ p.p.} \qquad \Delta Recall_{class\_2} = \frac{1}{10} = 10 \text{ p.p.}$$

Um único erro numa amostra de `class_2` derruba o recall dessa cultivar de 100% para 90%.

**Recall macro do Dummy:** como no treino, $(0 + 1 + 0)/3 = 33{,}33\%$, porque ele acerta só a `class_1`.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Modelos da Etapa 4, sem retreino | O teste avalia os modelos que seriam entregues |
| Matriz relativa no teste | Decisão do usuário, a mesma da referência |
| Célula simples para a matriz (e não a função de 3 versões) | No teste só há uma versão; a função da Etapa 6 desenha três |
| `target_names` e `zero_division=0` no relatório | Legibilidade e ausência de avisos (a referência gerava `UndefinedMetricWarning` no Dummy) |
| `digits=4` | Como na referência |
| Coluna de erros no resumo | Com 36 amostras, contar erros é mais honesto que comparar percentuais próximos |
| Ordem 5.1 SGD, 5.2 Dummy, 5.3 RF, 5.4 SVC | Mesma ordem da referência |

---

## 6. Interpretação dos resultados obtidos

**Resumo no teste (36 amostras):**

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) | Erros |
|---|---|---|---|---|---|
| Dummy | 38,89% | 12,96% | 33,33% | 18,67% | 22 |
| SGD | 94,44% | 95,05% | 94,29% | 94,53% | 2 |
| Random Forest | **100,00%** | **100,00%** | **100,00%** | **100,00%** | **0** |
| SVC (RBF) | 97,22% | 97,78% | 96,67% | 97,10% | 1 |

**Por classe:**

| Modelo | class_0 (P / R) | class_1 (P / R) | class_2 (P / R) |
|---|---|---|---|
| SGD | 92,31% / 100% | 92,86% / 92,86% | 100% / 90,00% |
| Random Forest | 100% / 100% | 100% / 100% | 100% / 100% |
| SVC | 100% / 100% | 93,33% / 100% | 100% / 90,00% |
| Dummy | 0% / 0% | 38,89% / 100% | 0% / 0% |

**Erros no teste:**
- SGD: 1 amostra de `class_1` → `class_0`; 1 de `class_2` → `class_1`;
- SVC: 1 amostra de `class_2` → `class_1`;
- Random Forest: nenhum;
- Dummy: as 12 de `class_0` e as 10 de `class_2` → `class_1`.

Os erros do SGD e do SVC foram listados numa verificação auxiliar fora do notebook. Eles também podem ser lidos nas matrizes relativas e nos relatórios.

**Leitura:**

1. **Dummy:** acerta só as 14 amostras de `class_1` (38,89%), com recall macro de 1/3.
2. **Random Forest acertou tudo** no teste; **SVC errou 1** e **SGD errou 2**.
3. **A ordem inverteu em relação à validação cruzada:** no treino fora da amostra, o SGD errou 1 de 142, o SVC 2 e a RF 3; no teste, RF 0, SVC 1 e SGD 2. Nos dois casos, as diferenças são de **1 ou 2 amostras**. Com 36 amostras no teste, cada erro vale 2,8 pontos, e a ordem entre os modelos muda conforme a divisão dos dados. A validação cruzada já indicava isso: cada modelo tinha dobras com 100% e dobras com erro.
4. **`class_2` no teste:** na validação cruzada ela nunca foi errada, mas no teste há uma amostra de `class_2` prevista como `class_1` pelo SGD e pelo SVC. Isso mostra que a fronteira `class_1`/`class_2`, já identificada como a mais difícil na Etapa 6, também aparece no teste, agora no sentido inverso.
5. **Conclusão desta etapa:** os três modelos reais generalizam muito bem (F1 macro de 94,5% a 100%), e não há base estatística, com 36 amostras, para afirmar que um é superior. A comparação completa (validação cruzada × teste) é feita na Etapa 8.

---

## 7. Erros comuns e limitações

- **Escolher o "melhor modelo" pelo teste:** com 36 amostras, a RF ter 0 erros e o SGD 2 é uma diferença de 2 amostras; escolher por isso seria usar o teste para decisão e superestimar a diferença.
- **Concluir que a RF é perfeita:** 100% em 36 amostras não garante 100% em dados novos (na validação cruzada ela errou 3 de 142).
- **Chamar `fit` no teste ou padronizar o teste à parte:** o Pipeline aplica só o `transform` do treino.
- **Esquecer `zero_division` no Dummy:** gera avisos (como na referência).
- **Limitação:** um único teste pequeno; outra divisão daria outra ordem entre os modelos.

---

## 8. Perguntas que o professor poderia fazer

1. **Qual modelo foi melhor no teste?** A Random Forest acertou as 36 amostras; o SVC errou 1 e o SGD 2. Mas são diferenças de 1 ou 2 amostras.
2. **Por que a ordem é diferente da validação cruzada?** Porque as diferenças são de poucas amostras: no treino o SGD errou menos, no teste a RF. Com conjuntos pequenos, isso é esperado.
3. **Então qual você escolheria?** Os três têm desempenho muito alto e próximo; os dados não sustentam a superioridade de um. A RF foi perfeita no teste, e o SGD foi o melhor na validação cruzada.
4. **O que o teste mostrou sobre as cultivares?** A fronteira entre `class_1` e `class_2` continua sendo a mais difícil: os únicos erros de `class_2` foram para `class_1`.
5. **Como a padronização foi aplicada no teste?** Pelo Pipeline, com a média e o desvio aprendidos no treino.
6. **Por que o recall macro do Dummy é 33,33%?** Ele acerta 100% da `class_1` e 0% das outras.

---

## 9. Como apresentar oralmente

> "Esta é a avaliação final, a única com o conjunto de teste: 36 vinhos, 12, 14 e 10 de cada cultivar. Os modelos estavam congelados, e o pipeline aplicou a padronização do treino. O Dummy acertou só a `class_1`, 39% de acurácia. A Random Forest acertou os 36 vinhos; o SVC errou 1 e o SGD errou 2, com F1 macro de 97% e 94,5%. Na validação cruzada, a ordem tinha sido diferente, com o SGD errando menos. Isso mostra que as diferenças entre esses três modelos são de uma ou duas amostras e dependem da divisão dos dados. A conclusão é que os três generalizam muito bem, e eu não afirmo que um seja superior. Os poucos erros continuam na fronteira entre `class_1` e `class_2`, a mesma que apareceu no treino."

Mostrar na tela: a tabela-resumo com o número de erros, o gráfico do teste e as matrizes relativas do SGD e do SVC.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Avaliação**

| Item | Para conferir | Link |
|---|---|---|
| `classification_report` | `target_names`, `digits`, `zero_division`, médias | [scikit-learn — classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |
| Guia do relatório | Exemplo comentado | [scikit-learn — Classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) |
| `accuracy_score` | Acurácia | [scikit-learn — accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) |
| `precision_score` | `average='macro'`, `zero_division` | [scikit-learn — precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) |
| `recall_score` | `average='macro'` | [scikit-learn — recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) |
| `f1_score` | `average='macro'`, `zero_division` | [scikit-learn — f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| `ConfusionMatrixDisplay` | `from_predictions`, `normalize`, `values_format` | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |
| `Pipeline` | `predict` com a transformação do treino | [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.join` | Métricas formatadas + número de erros | [pandas — DataFrame.join](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html) |
| `ax.bar_label` | Rótulos das barras | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
