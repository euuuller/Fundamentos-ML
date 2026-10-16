# Etapa 7 — *Trade-off* precisão × recall, curvas ROC e ajuste do limiar (F1)

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, subseção **Trade-off da Precisão e Recall, Curvas ROC e Ajuste do Limiar**, dentro da seção 4.
Correspondência na referência: subseção "Trade-off da Precisão e Recall" de `Docs/reference/1-Arquivo.ipynb` (`cross_val_predict` + `precision_recall_curve`). São acrescentadas as **curvas ROC** (exigidas pelo enunciado) e a escolha do limiar pelo **máximo F1** (adaptação A5 do plano). Estrutura igual à Etapa 6 do Exercício 01, que usava o F2.

---

## 1. Objetivo e relação com o exercício

1. Mostrar como o **limiar de decisão** troca falsos negativos por falsos positivos.
2. Desenhar as **curvas ROC** e calcular a **AUC** com previsões fora da amostra, para comparar os modelos independentemente do limiar.
3. Escolher, para o SGD e a RF, o limiar de **melhor equilíbrio** entre precisão e recall (máximo F1), porque no Titanic FN e FP têm importância equivalente.
4. Quantificar o efeito do novo limiar sobre FN, FP e as métricas.

Tudo usa **apenas o treino** (pontuações fora da amostra). O Dummy fica de fora, porque não ordena os passageiros (AUC = 0,5).

---

## 2. Código implementado, por blocos

### Bloco 1 — Pontuações fora da amostra

```python
y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=cv, method="decision_function")
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=cv, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(y_train, y_scores_sgd)
...
```

Igual ao Exercício 01, com o mesmo `cv`. O Pipeline inteiro (pré-processamento + modelo) é reajustado em cada dobra.

### Bloco 2 — Curvas ROC

```python
fpr_sgd, tpr_sgd, _ = roc_curve(y_train, y_scores_sgd)
auc_sgd = roc_auc_score(y_train, y_scores_sgd)
plt.plot(fpr_forest, tpr_forest, ...); plt.plot(fpr_sgd, tpr_sgd, ...); plt.plot([0, 1], [0, 1], "k:")
```

`roc_curve` devolve, para cada limiar, a FPR e a TPR. A diagonal pontilhada representa o classificador aleatório. A AUC vai na legenda.

### Bloco 3 — Limiar pelo máximo F1

```python
def f1_por_limiar(precisions, recalls):
    p, r = precisions[:-1], recalls[:-1]
    denominador = p + r
    return np.divide(2 * p * r, denominador, out=np.zeros_like(denominador), where=denominador > 0)

idx_sgd = f1_sgd.argmax()
limiar_sgd = thresholds_sgd[idx_sgd]
```

Mesma estrutura da função `f2_por_limiar` do Exercício 01, com a fórmula do F1. O `[:-1]` descarta o último ponto da curva (sem limiar), e `np.divide(..., where=...)` evita divisão por zero.

### Bloco 4 — Curva precisão × recall

As curvas da RF e do SGD com o **limiar padrão** (círculo) e o **limiar F1** (estrela). O ponto padrão é calculado diretamente das previsões no limiar padrão (`y_scores_sgd > 0`, `y_scores_forest > 0.5`).

### Bloco 5 — Precisão e recall × limiar

Um painel por modelo, com as linhas do limiar padrão e do limiar F1. Diferentemente do Exercício 01, não foi preciso zoom: as pontuações do SGD vão só de −17,91 a 20,51.

### Bloco 6 — Tabela padrão × F1

Para cada modelo e limiar: acurácia, precisão, recall, F1 e as **contagens de FN e FP**.

### Bloco 7 — Observações (Markdown)

Interpretação no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cross_val_predict(..., method="decision_function"/"predict_proba")` | Pontuações fora da amostra |
| `precision_recall_curve(y, scores)` | Precisão e recall para cada limiar |
| `roc_curve(y, scores)` | FPR e TPR para cada limiar |
| `roc_auc_score(y, scores)` | Área sob a curva ROC |
| `f1_por_limiar` | F1 de cada limiar |
| `limiar_sgd`, `limiar_forest` | **Limiares escolhidos, aplicados sem alteração no teste (Etapa 9)** |
| `y_scores_sgd`, `y_scores_forest` | Pontuações fora da amostra (reutilizadas na Etapa 8) |
| `pred_padrao_sgd`, `pred_padrao_forest` | Previsões no limiar padrão |
| `auc_sgd`, `auc_forest` | AUC fora da amostra (previsões agrupadas) |
| `df_limiares` | Tabela padrão × F1 |

---

## 4. Conceitos e fórmulas

**Regra de decisão:** $\hat{y} = 1$ se $s(\mathbf{x}) \geq t$. Baixar $t$ aumenta o recall e os FP; subir $t$ aumenta a precisão e os FN.

**Curva ROC:** pontos $(FPR(t), TPR(t))$ para todos os limiares $t$:

$$TPR = \frac{TP}{TP+FN} \qquad FPR = \frac{FP}{FP+TN}$$

**AUC** = área sob a curva = $P(s_{sobrevivente} > s_{não\ sobrevivente})$.

**ROC × precisão-recall:** a curva ROC usa os negativos (FPR) e, por isso, é pouco afetada pela proporção de classes. A curva precisão × recall foca nos positivos e mostra melhor o custo em precisão. As duas se complementam.

**F1 como critério:**

$$F1 = \frac{2PR}{P+R}$$

Precisão e recall têm o **mesmo peso**. Diferentemente do F2 do Exercício 01, que sempre puxa o limiar para baixo (mais recall), o F1 pode puxar o limiar **para qualquer lado**: para onde estiver o melhor equilíbrio do modelo.

**Por que a RF tem 234 limiares, e não no máximo 101:** com 100 árvores, cada uma com folhas puras, as probabilidades seriam múltiplos de 0,01. Mas 129 passageiros do treino têm atributos **idênticos** aos de outro passageiro, às vezes com desfecho diferente, e então algumas folhas não ficam puras. Isso gera probabilidades fracionárias (≈22% das pontuações não são múltiplos de 0,01; verificação auxiliar).

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Limiar pelo **máximo F1** | Decisão do usuário: no Titanic, FN e FP têm importância equivalente; F1 é o critério padrão de equilíbrio, sem meta arbitrária |
| Pontuações fora da amostra, mesmo `cv` | Sem usar o teste; coerência com a Etapa 6 |
| Curvas ROC no treino (fora da amostra) | Exigidas pelo enunciado; mostram a separação independentemente do limiar |
| Ponto padrão calculado das previsões | Mais direto e exato que procurar o índice na curva |
| Sem zoom nos gráficos | As escalas já permitem ver as diferenças (no Exercício 01 foi necessário) |
| Dummy fora | Não ordena os passageiros; qualquer limiar dá "todos 0" ou "todos 1" |

---

## 6. Interpretação dos resultados obtidos

**Pontuações e limiares:**

- SGD: pontuações de −17,91 a 20,51, com 681 limiares. **Limiar F1 = −0,8203** (padrão: 0).
- Random Forest: probabilidades de 0,00 a 1,00, com 234 limiares. **Limiar F1 = 0,5954** (padrão: 0,5).

**AUC fora da amostra (previsões agrupadas):** Random Forest **0,862** e SGD **0,787**, praticamente iguais às médias por dobra da Etapa 6 (86,31% e 78,75%).

**Comparação no treino (fora da amostra; 439 não sobreviventes e 273 sobreviventes):**

| Modelo - limiar | Acurácia | Precisão | Recall | F1 | FN | FP |
|---|---|---|---|---|---|---|
| SGD - padrão (0) | 71,91% | 61,89% | 69,60% | 65,52% | 83 | 117 |
| SGD - F1 (−0,820) | 72,05% | 60,57% | **77,66%** | **68,06%** | **61** | 138 |
| RF - padrão (0,5) | 78,51% | 70,83% | 74,73% | 72,73% | 69 | 84 |
| RF - F1 (0,595) | **80,76%** | **76,36%** | 72,16% | **74,20%** | 76 | **61** |

**Leitura:**

1. **Curvas ROC:** a curva da RF fica acima da do SGD em quase toda a extensão. A RF ordena melhor os passageiros em praticamente qualquer limiar.
2. **SGD:** o limiar F1 **desceu** (0 → −0,82). Foram identificados **22 sobreviventes a mais** (FN 83 → 61), ao custo de **21 alarmes falsos a mais** (FP 117 → 138). O F1 subiu 2,5 pontos.
3. **Random Forest:** o limiar F1 **subiu** (0,5 → 0,595). Foram **23 alarmes falsos a menos** (FP 84 → 61), ao custo de **7 sobreviventes a mais não identificados** (FN 69 → 76). O F1 subiu 1,5 ponto e a acurácia chegou a 80,76%.
4. **Por que direções opostas:** no limiar padrão, o SGD errava mais por FN do que precisava (recall baixo para sua precisão), e a RF errava mais por FP. O F1 corrigiu cada um na direção do seu desequilíbrio.
5. **Na curva P×R:** a estrela do SGD fica à direita do círculo (mais recall); a da RF fica à esquerda e acima (mais precisão).
6. **Nas curvas × limiar:** na RF, o limiar F1 fica perto do cruzamento das curvas de precisão e recall. No SGD, a curva de precisão fica irregular em limiares altos, porque poucos passageiros são previstos como sobreviventes ali e cada um muda muito a precisão.
7. **Sensibilidade (verificação auxiliar):** sem empates no máximo; perto dele, o F1 é quase plano (0,674–0,681 no SGD; 0,737–0,742 na RF). O ganho exato depende de poucos passageiros.

---

## 7. Erros comuns e limitações

- **Escolher o limiar no teste:** tornaria a avaliação final otimista. Não foi feito.
- **Usar pontuações "dentro da amostra"** (do modelo treinado no treino todo): seriam otimistas, sobretudo na RF.
- **Achar que ajustar o limiar sempre significa baixá-lo:** depende do critério; com F1, a RF subiu.
- **Confundir AUC com acurácia:** a AUC independe do limiar; mudar o limiar não altera a curva ROC, só a posição do modelo sobre ela.
- **Limitação:** o limiar escolhido coincide com a pontuação de um passageiro do treino; em dados novos, o ganho pode ser menor (ou nulo, como ocorreu com a RF no Exercício 01).
- **Limitação:** no `cross_val_predict`, as pontuações de cada dobra vêm de modelos diferentes. No SGD, a escala da `decision_function` pode variar entre eles, como no Exercício 01.

---

## 8. Perguntas que o professor poderia fazer

1. **O que a curva ROC mostra?** A taxa de verdadeiros positivos contra a de falsos positivos para todos os limiares; quanto mais perto do canto superior esquerdo, melhor.
2. **Qual modelo tem a melhor curva?** A Random Forest (AUC 0,862 contra 0,787), acima do SGD em quase toda a curva.
3. **Por que F1 e não F2, como no câncer?** Aqui não há um erro claramente mais grave; o F1 dá peso igual a precisão e recall.
4. **Por que o limiar da RF subiu e o do SGD desceu?** O F1 busca o equilíbrio. No limiar padrão, o SGD tinha FN demais e a RF tinha FP demais, e cada ajuste corrigiu o desequilíbrio do próprio modelo.
5. **Qual o efeito no SGD?** +22 sobreviventes identificados, +21 alarmes falsos, F1 de 65,5% para 68,1%.
6. **E na RF?** −23 alarmes falsos, +7 sobreviventes perdidos, F1 de 72,7% para 74,2%.
7. **Mudar o limiar altera a AUC?** Não. A AUC considera todos os limiares; o limiar só escolhe um ponto da curva.

---

## 9. Como apresentar oralmente

> "As curvas ROC mostram o desempenho em todos os limiares de uma vez. A Random Forest fica acima do SGD em quase toda a curva, com AUC de 0,86 contra 0,79, e as duas bem acima do sorteio. Depois escolhi o limiar de cada modelo pelo F1, que equilibra precisão e recall, já que aqui errar para um lado ou para o outro tem o mesmo peso. Usei só previsões fora da amostra do treino. O interessante é que os limiares se moveram em direções opostas. No SGD, o limiar desceu: ele passou a identificar 22 sobreviventes a mais, com 21 alarmes falsos a mais. Na Random Forest, o limiar subiu: ela cortou 23 alarmes falsos e perdeu 7 sobreviventes. Nos dois, o F1 melhorou. Esses limiares ficam fixos e serão testados no conjunto de teste."

Mostrar na tela: as curvas ROC, a curva P×R com círculos e estrelas e a tabela FN/FP.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Pontuações e curvas**

| Item | Para conferir | Link |
|---|---|---|
| `cross_val_predict` | Parâmetro `method` | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |
| `precision_recall_curve` | Formato do retorno | [scikit-learn — precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) |
| `roc_curve` | FPR, TPR e limiares | [scikit-learn — roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) |
| `roc_auc_score` | Cálculo da AUC | [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) |
| Guia ROC | Conceitos de ROC e AUC | [scikit-learn — Receiver operating characteristic (ROC)](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) |
| Exemplo oficial ROC | Curvas comentadas | [scikit-learn — ROC example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) |

**Limiar e métricas**

| Item | Para conferir | Link |
|---|---|---|
| Ajuste do limiar | Guia oficial | [scikit-learn — Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) |
| `TunedThresholdClassifierCV` | Alternativa automática, citada e não usada | [scikit-learn — TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) |
| `f1_score` | Definição | [scikit-learn — f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| `confusion_matrix` | Ordem TN, FP, FN, TP no `ravel()` | [scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| Árvores de decisão | Folhas impuras e probabilidades | [scikit-learn — Decision Trees](https://scikit-learn.org/stable/modules/tree.html) |

**NumPy e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `np.divide` com `where`/`out` | Divisão segura | [NumPy — numpy.divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) |
| `ax.axvline` | Linhas dos limiares | [Matplotlib — Axes.axvline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html) |
