# Etapa 9 — Avaliação final no teste

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **5. Teste** (5.1 SGD, 5.2 Dummy, 5.3 Random Forest, 5.4 Curvas ROC no Teste, 5.5 Resumo do Teste).
Correspondência na referência: seção 5 de `Docs/reference/1-Arquivo.ipynb` (relatório, métricas e matriz por modelo; resumo). Foram acrescentados a AUC, os limiares F1 e a subseção **5.4 Curvas ROC no Teste**; por isso o resumo passou a ser a 5.5.

---

## 1. Objetivo e relação com o exercício

Esta é a **avaliação final**: medir o desempenho com passageiros que **não participaram de nenhuma decisão**. O conjunto de teste (179 passageiros, 69 sobreviventes) é usado **aqui pela única vez**.

| Variante | Modelo (Etapa 5) | Limiar |
|---|---|---|
| Dummy | `dummy_clf` | — (sempre Não Sobreviveu) |
| SGD - padrão | `sgd_clf` | 0 |
| SGD - F1 | `sgd_clf` | `limiar_sgd` = −0,820 (Etapa 7) |
| Random Forest - padrão | `forest_clf` | 0,5 |
| Random Forest - F1 | `forest_clf` | `limiar_forest` = 0,595 (Etapa 7) |

Para cada variante: `classification_report`, as 5 métricas do enunciado (acurácia, precisão, recall, F1, AUC) e **uma matriz de confusão própria**; depois, as **curvas ROC no teste** e um resumo.

---

## 2. Código implementado, por blocos

### Bloco 1 — Introdução e função de métricas

```python
def metricas_teste(y_pred, y_score):
    resultado = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_score),
    }
    ...
```

Igual à do Exercício 01, com um segundo argumento, `y_score`, porque a **AUC precisa das pontuações contínuas**, e não das classes previstas.

### Bloco 2 — Cada variante (5.1, 5.2, 5.3)

```python
y_scores_teste_sgd = sgd_clf.decision_function(X_test)
y_pred_sgd = sgd_clf.predict(X_test)
print(classification_report(y_test, y_pred_sgd, target_names=nomes_classes_relatorio))
teste_sgd = metricas_teste(y_pred_sgd, y_scores_teste_sgd)
cm_teste_sgd = plot_matriz_confusao(y_test, y_pred_sgd, "SGD - limiar padrão (0) - Teste")
```

- O **Pipeline** recebe `X_test` bruto, com texto e ausentes, e aplica o pré-processamento **aprendido no treino** antes de pontuar e classificar.
- As versões F1 aplicam o limiar da Etapa 7: `(y_scores_teste_sgd >= limiar_sgd).astype(int)`.
- O Dummy usa `predict_proba(X_test)[:, 1]` como pontuação, a mesma para todos os passageiros, o que dá AUC = 0,5.

### Bloco 3 — Curvas ROC no teste (5.4)

```python
fpr_teste_sgd, tpr_teste_sgd, _ = roc_curve(y_test, y_scores_teste_sgd)
...
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
plt.plot(fp / (fp + tn), tp / (tp + fn), marcador, ...)
```

Desenha as curvas do SGD e da RF e marca **onde cada limiar caiu** sobre a curva: círculo para o padrão e estrela para o F1. O ponto é calculado com $FPR = FP/(FP+TN)$ e $TPR = TP/(TP+FN)$.

### Bloco 4 — Resumo (5.5)

Tabela com as 5 métricas, as contagens TN/FP/FN/TP e o total de erros de cada variante, e gráfico de barras no estilo da referência (`tab10`).

### Bloco 5 — Observações (Markdown)

Interpretação no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `classification_report(..., target_names=...)` | Precisão, recall, F1 e *support* por classe; acurácia; médias macro e ponderada |
| `metricas_teste(y_pred, y_score)` | Calcula, imprime e retorna as 5 métricas |
| `sgd_clf.decision_function(X_test)` | Pontuações do SGD no teste |
| `forest_clf.predict_proba(X_test)[:, 1]` | Probabilidade de Sobreviveu (RF) |
| `dummy_clf.predict_proba(X_test)[:, 1]` | Pontuação constante (Dummy) |
| `roc_curve(y_test, scores)` | Curva ROC no teste |
| `teste_*`, `cm_teste_*` | Métricas e matrizes de cada variante |
| `df_teste` | Tabela-resumo do teste |

---

## 4. Conceitos e fórmulas

**`classification_report`:** a linha **Sobreviveu (1)** traz as métricas do exercício. O recall da linha **Não Sobreviveu (0)** é a especificidade. *support* é o número real de passageiros de cada classe (110 e 69).

**AUC × limiar:** a AUC usa todas as pontuações e não depende do limiar. Por isso é **igual** nas versões padrão e F1 do mesmo modelo. Mudar o limiar só desloca o ponto de operação **sobre** a mesma curva ROC (círculo → estrela).

**Granularidade do teste:**

$$\Delta Recall = \frac{1}{69} \approx 1{,}45 \text{ p.p.} \qquad \Delta Especificidade = \frac{1}{110} \approx 0{,}91 \text{ p.p.}$$

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Modelos da Etapa 5 sem retreino | O teste mede o modelo que seria entregue |
| Limiares da Etapa 7 sem alteração | A especificação proíbe escolher limiares com o teste |
| AUC no resumo e nas curvas | Exigida pelo enunciado |
| Pontos dos limiares sobre a curva ROC | Liga a curva (independente do limiar) às métricas de cada limiar |
| Matriz própria para cada variante, com siglas | Especificação e pedido do usuário |
| Coluna de total de erros | FN e FP têm o mesmo peso neste exercício |
| Subseção 5.4 nova (ROC) e resumo em 5.5 | A referência não tem ROC; o resumo continua sendo a última subseção, como na referência |

---

## 6. Interpretação dos resultados obtidos

**Resumo no teste (110 não sobreviventes, 69 sobreviventes):**

| Modelo - limiar | Acurácia | Precisão | Recall | F1 | AUC | TN | FP | FN | TP | Erros |
|---|---|---|---|---|---|---|---|---|---|---|
| Dummy | 61,45% | 0,00% | 0,00% | 0,00% | 50,00% | 110 | 0 | **69** | 0 | 69 |
| SGD - padrão | 79,33% | 78,57% | 63,77% | 70,40% | **83,70%** | 98 | 12 | 25 | 44 | 37 |
| SGD - F1 | 75,98% | 67,57% | 72,46% | 69,93% | **83,70%** | 86 | 24 | 19 | 50 | 43 |
| RF - padrão | 81,01% | 76,92% | 72,46% | 74,63% | 83,04% | 95 | 15 | 19 | 50 | 34 |
| RF - F1 | **82,12%** | **80,33%** | 71,01% | **75,38%** | 83,04% | 98 | 12 | 20 | 49 | **32** |

**Leitura:**

1. **Dummy:** não identifica nenhum dos 69 sobreviventes; a acurácia de 61,45% é a proporção de não sobreviventes (110/179), e a AUC é 50%.
2. **Melhor configuração no teste:** **Random Forest com limiar F1**, com maior acurácia (82,12%), precisão (80,33%) e F1 (75,38%) e menos erros (32).
3. **Ajuste F1 na RF:** trocou **3 FP por 1 FN** (FP 15 → 12; FN 19 → 20), e o F1 subiu de 74,63% para 75,38%. É o mesmo sentido do treino (Etapa 7), com efeito menor.
4. **Ajuste F1 no SGD:** +6 sobreviventes identificados (FN 25 → 19), mas **FP dobrou** (12 → 24). O F1 caiu levemente (70,40% → 69,93%) e o total de erros subiu (37 → 43). **No teste, o ajuste não melhorou o SGD**, ao contrário do treino.
5. **AUC: inversão em relação ao treino.** No teste, SGD (83,70%) e RF (83,04%) ficaram praticamente empatados, com as curvas ROC se cruzando várias vezes. Na validação cruzada, a RF tinha AUC claramente maior (86,31% × 78,75%). O SGD foi **melhor** no teste do que na validação cruzada (F1 70,40% contra 65,33%), e a RF teve AUC **menor** (83,04% contra 86,31%).
6. **Por que isso pode acontecer:** o teste tem só 179 passageiros, e o SGD já se mostrava **instável** na validação cruzada (F1 entre 56% e 71% conforme a dobra). O F1 do SGD no teste (70,40%) está dentro dessa faixa. Um único teste pequeno não permite separar sorte de diferença real. A validação cruzada, com 5 avaliações, é a evidência mais estável de que a RF ordena melhor.
7. **Diferenças em passageiros:** entre RF-F1 e RF-padrão, a diferença é de 2 erros; entre RF-F1 e SGD-padrão, de 5 erros. São números pequenos.

---

## 7. Erros comuns e limitações

- **Ajustar o limiar depois de ver o teste** ("para o SGD, o limiar 0 era melhor"): transformaria o teste em validação. Não foi feito.
- **Calcular a AUC com as classes previstas:** a AUC precisa de `y_score` contínuo, e a função recebe as pontuações separadamente.
- **Concluir que o SGD "é melhor em AUC" com base no teste:** a diferença (0,66 p.p.) é pequena e contraria a validação cruzada; não é conclusiva.
- **Limitação:** um único teste de 179 passageiros; outra divisão daria números diferentes.
- **Limitação:** hiperparâmetros padrão; o SGD, em particular, é sensível a eles.

---

## 8. Perguntas que o professor poderia fazer

1. **Qual a melhor configuração no teste?** Random Forest com limiar F1: F1 de 75,38%, acurácia de 82,12% e 32 erros.
2. **O ajuste do limiar funcionou?** Na RF, sim (−3 FP, +1 FN, F1 melhor). No SGD, não: identificou 6 sobreviventes a mais, mas dobrou os FP, e o F1 caiu um pouco.
3. **Por que a AUC é igual nas versões padrão e F1?** Porque a AUC não depende do limiar; o limiar só escolhe um ponto da curva.
4. **O SGD teve AUC maior que a RF no teste. Então ele é melhor?** Não se pode concluir. A diferença é pequena, as curvas se cruzam, e na validação cruzada a RF foi melhor nas 5 dobras. O teste é pequeno, e o SGD é instável.
5. **O Pipeline tratou os ausentes do teste?** Sim: os 40 passageiros sem idade receberam a mediana aprendida no treino (28,5).
6. **Por que não ajustar nada agora?** Qualquer ajuste baseado no teste tornaria a avaliação final otimista.

---

## 9. Como apresentar oralmente

> "Esta é a avaliação final, a única com o conjunto de teste: 179 passageiros, 69 sobreviventes. Os pipelines receberam os dados brutos e aplicaram o pré-processamento aprendido no treino. O Dummy não identificou nenhum sobrevivente. A melhor configuração foi a Random Forest com o limiar escolhido pelo F1: 82% de acurácia, F1 de 75% e só 32 erros; o ajuste trocou 3 alarmes falsos por 1 sobrevivente perdido, como no treino. No SGD, o ajuste não funcionou no teste: identificou 6 sobreviventes a mais, mas dobrou os alarmes falsos. Um ponto importante: no teste, a AUC do SGD e da Random Forest ficou praticamente empatada, perto de 0,83, enquanto na validação cruzada a Random Forest era claramente melhor. Com um teste pequeno e um SGD instável, não tiro conclusões dessa inversão; a validação cruzada é a evidência mais estável."

Mostrar na tela: as curvas ROC no teste (com círculos e estrelas), as matrizes da RF-F1 e do SGD-F1 e a tabela-resumo.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Avaliação**

| Item | Para conferir | Link |
|---|---|---|
| `classification_report` | `target_names`, `zero_division`, médias | [scikit-learn — classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |
| Guia do relatório | Exemplo comentado | [scikit-learn — Classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) |
| `roc_auc_score` | AUC com pontuações contínuas | [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) |
| `roc_curve` | Curva ROC no teste | [scikit-learn — roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) |
| `Pipeline` | `predict`/`decision_function`/`predict_proba` com pré-processamento | [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| `DummyClassifier` | `predict_proba` constante | [scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) |
| `ConfusionMatrixDisplay` | Matrizes do teste (ver guia 08) | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.join` | Métricas formatadas + contagens | [pandas — DataFrame.join](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html) |
| `ax.bar_label` | Rótulos das barras | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
