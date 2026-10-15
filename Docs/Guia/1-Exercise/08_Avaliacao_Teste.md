# Etapa 8 — Avaliação final no teste, com matrizes de confusão separadas

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **5. Teste** (5.1 SGD, 5.2 Dummy, 5.3 Random Forest, 5.4 Resumo do Teste).
Correspondência na referência: seção 5 de `Docs/reference/1-Arquivo.ipynb` (`classification_report`, 4 métricas e matriz por modelo; gráfico-resumo em 5.4). Foram acrescentados o limiar F2 do SGD e da RF, uma tabela-resumo com as contagens e a correção do erro `X_test_array` da referência.

---

## 1. Objetivo e relação com o exercício

Esta é a **avaliação final**: medir como os modelos se comportam com **tumores que nunca participaram de nenhuma decisão**. O conjunto de teste (114 tumores, 42 malignos), separado na Etapa 3, é usado **aqui pela primeira e única vez**.

O que é avaliado, sem nenhum ajuste:

| Variante | Modelo (Etapa 4) | Limiar |
|---|---|---|
| Dummy | `dummy_clf` | — (sempre Benigno) |
| SGD - padrão | `sgd_clf` | 0 |
| SGD - F2 | `sgd_clf` | `limiar_sgd` = −2,929 (Etapa 6) |
| Random Forest - padrão | `forest_clf` | 0,5 |
| Random Forest - F2 | `forest_clf` | `limiar_forest` = 0,45 (Etapa 6) |

---

## 2. Código implementado, por blocos

### Bloco 1 — Introdução (Markdown) e função de métricas

```python
from sklearn.metrics import classification_report, accuracy_score

nomes_classes_relatorio = ['Benigno (0)', 'Maligno (1)']

def metricas_teste(y_pred):
    resultado = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'F2-Score': fbeta_score(y_test, y_pred, beta=2, zero_division=0),
    }
    for nome, valor in resultado.items():
        print(f"{nome:9s} {valor:.2%}")
    return resultado
```

Na referência, as 4 linhas de `print` com as métricas se repetiam em cada modelo. Aqui são **5 variantes**, então o cálculo virou uma função que **imprime** (como a referência) e **retorna** os valores para a tabela-resumo. O F2 foi incluído para manter coerência com o critério da Etapa 6.

### Bloco 2 — Cada modelo no limiar padrão (5.1, 5.2, 5.3)

```python
y_pred_sgd = sgd_clf.predict(X_test)
print(classification_report(y_test, y_pred_sgd, target_names=nomes_classes_relatorio))
teste_sgd = metricas_teste(y_pred_sgd)
cm_teste_sgd = plot_matriz_confusao(y_test, y_pred_sgd, "SGD - limiar padrão (0) - Teste")
```

- `sgd_clf.predict(X_test)`: o **Pipeline** aplica ao teste o `transform` do `StandardScaler` com a média e o desvio **aprendidos no treino** e depois classifica. Na referência, esta célula usava `X_test_array`, uma variável inexistente; com o Pipeline, basta `X_test`.
- O Dummy e a RF recebem `X_test` diretamente, como na referência.
- `target_names` substitui "0"/"1" por "Benigno (0)"/"Maligno (1)" no relatório.
- Cada variante tem **sua própria matriz**, desenhada com a função da Etapa 7, já com as siglas TN/FP/FN/TP.

### Bloco 3 — SGD e RF no limiar F2

```python
y_scores_teste_sgd = sgd_clf.decision_function(X_test)
y_pred_sgd_f2 = (y_scores_teste_sgd >= limiar_sgd).astype(int)

y_scores_teste_forest = forest_clf.predict_proba(X_test)[:, 1]
y_pred_forest_f2 = (y_scores_teste_forest >= limiar_forest).astype(int)
```

Calcula a pontuação de cada tumor do teste e aplica o limiar **escolhido no treino**. Os limiares não são recalculados.

### Bloco 4 — Resumo do teste (5.4)

```python
df_teste = pd.DataFrame({modelo: metricas for modelo, (metricas, cm) in resultados_teste.items()}).T
df_teste[['TN', 'FP', 'FN', 'TP']] = [cm.ravel() for metricas, cm in resultados_teste.values()]
```

Junta as métricas e as contagens das 5 variantes numa tabela. O gráfico de barras segue o estilo da seção 5.4 da referência (`colormap="tab10"`, valores sobre as barras), com as **4 métricas do enunciado**. Os rótulos ficaram na vertical porque são 20 barras.

### Bloco 5 — Observações (Markdown)

Interpretação dos resultados no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `classification_report(y_test, y_pred, target_names=...)` | Precisão, recall, F1 e suporte **por classe**, mais acurácia, média macro e média ponderada |
| `zero_division=0` | No Dummy, a precisão da classe Maligno é 0/0; o parâmetro evita o aviso e define 0 |
| `metricas_teste(y_pred)` | Função do notebook: calcula, imprime e retorna as 5 métricas |
| `sgd_clf.decision_function(X_test)` | Pontuação do SGD para o teste (com padronização do treino) |
| `forest_clf.predict_proba(X_test)[:, 1]` | Probabilidade de Maligno, segundo a RF |
| `y_pred_*`, `y_pred_*_f2` | Previsões no teste (padrão e F2) |
| `teste_*` | Dicionários com as métricas de cada variante |
| `cm_teste_*` | Matrizes de confusão do teste |
| `df_teste` | Tabela-resumo (métricas + TN/FP/FN/TP) |

---

## 4. Conceitos e fórmulas

**Leitura do `classification_report`:**

- **support:** quantidade real de cada classe no teste (72 benignos, 42 malignos).
- Linha **Maligno (1)**: precisão, recall e F1 da classe positiva, as métricas do exercício.
- Linha **Benigno (0)**: o recall dessa linha é a **especificidade**, $TN/(TN+FP)$.
- **macro avg:** média simples das duas classes (cada classe pesa igual).
- **weighted avg:** média ponderada pelo *support* (a classe maior pesa mais).

**Granularidade no teste:**

$$\Delta Recall = \frac{1}{42} \approx 2{,}38 \text{ p.p. por maligno} \qquad \Delta Especificidade = \frac{1}{72} \approx 1{,}39 \text{ p.p. por benigno}$$

**Por que o teste vale mais que o treino para a conclusão:** nenhuma escolha (modelo, padronização, limiar) usou estes dados. Mas, por ser **um único conjunto pequeno**, a estimativa tem variância alta, e outra divisão poderia dar números diferentes.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Usar os modelos da Etapa 4, sem retreinar | Eles foram treinados com todo o treino; retreinar com treino + teste impediria a avaliação |
| Aplicar os limiares da Etapa 6 sem alterar | A especificação proíbe usar o teste para escolher limiares |
| Avaliar o limiar padrão **e** o F2 | Mostra, em dados novos, se o ajuste orientado a recall se confirma |
| Uma matriz por variante, com siglas | Exigência da especificação (matrizes separadas, classes e eixos identificados) e pedido do usuário (siglas) |
| `target_names` no relatório | Deixa claro qual linha é Maligno |
| Função `metricas_teste` | 5 repetições do mesmo cálculo; a função mantém o formato da referência |
| Ordem 5.1 SGD, 5.2 Dummy, 5.3 RF | Mesma ordem da referência |

---

## 6. Interpretação dos resultados obtidos

**Resumo no teste (72 benignos, 42 malignos):**

| Modelo - limiar | Acurácia | Precisão | Recall | F1 | F2 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Dummy | 63,16% | 0,00% | 0,00% | 0,00% | 0,00% | 72 | 0 | **42** | 0 |
| SGD - padrão | 92,11% | 86,67% | 92,86% | 89,66% | 91,55% | 66 | 6 | **3** | 39 |
| SGD - F2 | 92,98% | 86,96% | **95,24%** | 90,91% | **93,46%** | 66 | 6 | **2** | 40 |
| Random Forest - padrão | **97,37%** | **100,00%** | 92,86% | **96,30%** | 94,20% | 72 | 0 | **3** | 39 |
| Random Forest - F2 | 96,49% | 97,50% | 92,86% | 95,12% | 93,75% | 71 | 1 | **3** | 39 |

**Leitura:**

1. **Dummy:** os 42 malignos do teste seriam perdidos. A acurácia de 63,16% é a proporção de benignos no teste (72/114). Mais uma vez, a acurácia esconde o fracasso.
2. **SGD, limiar padrão → F2:** **1 câncer a mais detectado** (FN 3 → 2; recall 92,86% → 95,24%) **sem nenhum alarme falso adicional** (FP 6 → 6). No teste, o ajuste orientado a recall funcionou para o SGD.
3. **Random Forest, limiar padrão → F2:** o recall **não mudou** (3 FN nos dois casos), e surgiu 1 FP (0 → 1). No teste, o ajuste do limiar **não trouxe benefício** para a RF.
4. **Melhor recall:** SGD - F2 (40/42). **Melhor acurácia, precisão e F1:** Random Forest no limiar padrão (sem nenhum falso positivo).
5. **Diferenças pequenas:** todos os modelos reais ficaram entre 2 e 3 FN; a diferença é de **1 tumor** (≈2,4 p.p. de recall). Com 42 malignos, isso **não permite afirmar** que um modelo é superior.
6. **SGD × RF nos falsos positivos:** no teste, o SGD gerou 6 alarmes falsos (precisão ≈87%), e a RF, 0 ou 1. É uma diferença maior que a observada no treino. Comparar treino e teste em detalhe é assunto da Etapa 9.

**Análise auxiliar das pontuações** (fora do notebook, feita **depois** da avaliação, apenas para explicar os resultados, sem alterar nada):

- **SGD:** um dos malignos do teste teve pontuação **−0,04**, logo abaixo do limiar padrão (0). O limiar F2 (−2,93) passou a detectá-lo, e **nenhum benigno** do teste tinha pontuação entre −2,93 e 0. Por isso o recall subiu sem aumentar os FP. Os outros 2 malignos não detectados tinham pontuações muito mais baixas (≈ −16 e −15), que nenhum ajuste razoável alcançaria.
- **Random Forest:** os 3 malignos não detectados tinham probabilidades de **0,28, 0,37 e 0,41**, todas abaixo de 0,45, e o novo limiar não os alcançou. Um benigno com probabilidade **0,47** passou a ser FP.
- **Os modelos erram em tumores diferentes:** só 1 dos malignos não detectados é comum ao SGD - F2 e à RF. Os 6 FP do SGD receberam probabilidades baixas da RF (≤ 0,28).

---

## 7. Erros comuns e limitações

- **Ajustar algo depois de ver o teste** (por exemplo, "o limiar 0,40 seria melhor para a RF no teste"): transformaria o teste em validação e invalidaria a estimativa. Não foi feito.
- **Chamar `fit` ou `fit_transform` no teste:** o Pipeline aplica apenas `transform` com os parâmetros do treino.
- **Usar uma variável inexistente** (`X_test_array` na referência): corrigido usando `X_test`.
- **Ler a linha errada do relatório:** as métricas do exercício estão na linha **Maligno (1)**; a linha Benigno descreve a outra classe.
- **Limitação, teste pequeno:** 42 malignos e 72 benignos. Cada tumor muda o recall em ≈2,4 p.p. e a especificidade em ≈1,4 p.p.
- **Limitação, um único *split*:** a variação entre divisões, vista na validação cruzada (Etapa 5), indica que outra divisão de teste poderia dar números diferentes.
- **Limitação, contexto acadêmico:** os resultados **não têm validade clínica**; são de um exercício com um dataset público de 1995.

---

## 8. Perguntas que o professor poderia fazer

1. **O teste foi usado antes?** Não. Foi separado na Etapa 3 e usado só aqui; modelos e limiares estavam fixos.
2. **Como a padronização foi aplicada no teste?** O Pipeline aplicou o `transform` com a média e o desvio aprendidos no treino.
3. **O ajuste do limiar funcionou?** No SGD, sim: 1 câncer a mais detectado, sem novos alarmes falsos. Na RF, não: o recall ficou igual e surgiu 1 FP.
4. **Por que não ajustar o limiar da RF agora que se sabe o resultado?** Porque seria usar o teste para escolher o limiar, e a avaliação deixaria de ser imparcial.
5. **Qual modelo você escolheria?** Pelos critérios do exercício, o SGD - F2 teve o maior recall (95,24%), e a RF padrão teve a melhor precisão e acurácia. A diferença de recall é de 1 tumor, então a escolha depende do custo atribuído a FN e FP, e a conclusão deve ser cautelosa.
6. **O que o support do relatório significa?** O número real de tumores de cada classe no teste (72 benignos e 42 malignos).
7. **Por que a acurácia do Dummy é 63,16%?** É a proporção de benignos no teste (72/114).

---

## 9. Como apresentar oralmente

> "Esta é a avaliação final, a única em que usei o conjunto de teste: 114 tumores, 42 malignos. Os modelos e os limiares estavam congelados. O Dummy perde os 42 malignos, apesar de 63% de acurácia. O SGD no limiar padrão errou 3 malignos; com o limiar F2 escolhido no treino, errou só 2, sem aumentar os alarmes falsos, e chegou a 95,2% de recall, o maior do teste. A Random Forest no limiar padrão teve a maior acurácia, 97,4%, e nenhum falso positivo, mas também perdeu 3 malignos; o ajuste do limiar não mudou o recall dela e acrescentou 1 falso positivo. Como as diferenças de recall são de um tumor só, não afirmo que um modelo é superior. E não ajustei nada depois de ver o teste, para não contaminar a avaliação."

Mostrar na tela: as duas matrizes do SGD (padrão e F2), a matriz da RF padrão e a tabela-resumo.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Avaliação**

| Item | Para conferir | Link |
|---|---|---|
| `classification_report` | Parâmetros `target_names`, `digits`, `zero_division` e o significado de *macro*/*weighted avg* | [scikit-learn — classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |
| Guia do relatório | Exemplo comentado | [scikit-learn — Classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) |
| `accuracy_score` | Acurácia | [scikit-learn — accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) |
| `fbeta_score` | F2 (β = 2) | [scikit-learn — fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) |
| `Pipeline` | Como `predict`/`decision_function` aplicam o `transform` antes do modelo | [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| `ConfusionMatrixDisplay` e `confusion_matrix` | Matrizes do teste (ver também o guia 07) | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.T` | Transpor a tabela (modelos nas linhas) | [pandas — DataFrame.T](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.T.html) |
| `DataFrame.join` | Juntar as métricas formatadas às contagens | [pandas — DataFrame.join](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html) |
| `ax.bar_label` | Rótulos de valor (aqui com `rotation=90`) | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
