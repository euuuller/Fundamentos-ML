# Etapa 6 — *Trade-off* entre precisão e recall e ajuste do limiar

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, subseção **Trade-off da Precisão e Recall**, dentro da seção 4, depois do gráfico da validação cruzada.
Correspondência na referência: subseção "Trade-off da Precisão e Recall" de `Docs/reference/1-Arquivo.ipynb` (`cross_val_predict` + `precision_recall_curve`, curvas do SGD e da Random Forest). A escolha do limiar por F2 é uma adaptação: a referência só marcava um limiar ilustrativo (3000).

---

## 1. Objetivo e relação com o exercício

O título do exercício é **"Otimização de Recall"**. Os modelos da Etapa 5 usavam o limiar padrão. Esta etapa:

1. mostra que o **limiar de decisão** controla o equilíbrio entre falsos negativos (FN) e falsos positivos (FP);
2. escolhe, para o SGD e a Random Forest, um limiar **orientado a recall** com um critério explícito (F2);
3. quantifica **quantos cânceres a mais são detectados** e **quantos alarmes falsos a mais** isso custa.

Tudo é feito com **previsões fora da amostra do treino**; o teste não é usado. O Dummy fica de fora porque não tem pontuação informativa: ele prevê sempre a mesma classe.

---

## 2. Código implementado, por blocos

### Bloco 1 — Pontuações fora da amostra

```python
y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=cv, method="decision_function")
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=cv, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]

precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(y_train, y_scores_sgd)
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train, y_scores_forest)
```

- `cross_val_predict` usa o **mesmo `cv`** da Etapa 5. Cada tumor do treino recebe a pontuação do modelo treinado nas outras 4 dobras, ou seja, de um modelo que não o viu.
- `method="decision_function"` devolve a pontuação contínua do SGD; `method="predict_proba"` devolve as probabilidades da RF, e a coluna `[:, 1]` é a probabilidade de **Maligno**.
- `precision_recall_curve` calcula precisão e recall para **cada limiar possível** (cada valor distinto de pontuação).

### Bloco 2 — Escolha do limiar por F2

```python
def f2_por_limiar(precisions, recalls):
    p, r = precisions[:-1], recalls[:-1]
    denominador = 4 * p + r
    return np.divide(5 * p * r, denominador, out=np.zeros_like(denominador), where=denominador > 0)

f2_sgd = f2_por_limiar(precisions_sgd, recalls_sgd)
idx_sgd = f2_sgd.argmax()
limiar_sgd = thresholds_sgd[idx_sgd]
```

- A função existe porque o mesmo cálculo é feito para dois modelos, o tipo de repetição que a especificação permite eliminar.
- `[:-1]`: `precision_recall_curve` devolve um ponto final (precisão = 1, recall = 0) **sem limiar associado**; ele é descartado para alinhar os arrays.
- `np.divide(..., where=denominador > 0)` evita divisão por zero.
- `argmax()` devolve a posição do maior F2, e o limiar nessa posição é o escolhido.

### Bloco 3 — Curva Precisão × Recall (visão completa + zoom)

Mesmo gráfico da referência (curvas da RF e do SGD sobrepostas), em dois painéis. À esquerda, a curva inteira; à direita, um zoom entre 85% e 100%, onde ficam os pontos de interesse. Círculos marcam o limiar padrão e estrelas, o limiar F2.

### Bloco 4 — Precisão e recall em função do limiar

Um painel por modelo, com as curvas de precisão (tracejada) e recall (contínua), a linha do limiar padrão (pontilhada) e a do limiar F2 (traço-ponto). No SGD, o eixo X foi limitado a [−40, 40], porque as pontuações vão de cerca de −147 a 509 e, sem zoom, as duas linhas se sobrepõem.

### Bloco 5 — Tabela comparativa (padrão × ajustado)

```python
previsoes_limiar = {
    'SGD - padrão (0)': y_scores_sgd > 0,
    f'SGD - F2 ({limiar_sgd:.3f})': y_scores_sgd >= limiar_sgd,
    'Random Forest - padrão (0,5)': y_scores_forest > 0.5,
    f'Random Forest - F2 ({limiar_forest:.3f})': y_scores_forest >= limiar_forest,
}
...
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
```

Para cada modelo e limiar: precisão, recall, F1, F2 e as **contagens** de FN e FP. O `.ravel()` "achata" a matriz 2×2 na ordem TN, FP, FN, TP. A comparação usa `>` no padrão, que é o comportamento do `predict` (o SGD prevê 1 se a pontuação for > 0; a RF, se a probabilidade for > 0,5), e `>=` no limiar F2, a convenção do `precision_recall_curve`.

### Bloco 6 — Observações (Markdown)

Interpretação dos números no próprio notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cross_val_predict(modelo, X, y, cv=cv, method=...)` | Uma previsão por amostra, cada uma feita por um modelo que não a viu |
| `method="decision_function"` | Pontuação contínua do SGD: $s(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$ |
| `method="predict_proba"` | Probabilidades por classe (colunas na ordem das classes: 0, 1) |
| `precision_recall_curve(y, scores)` | Retorna `precisions`, `recalls` (n+1 valores) e `thresholds` (n valores, em ordem crescente) |
| `f2_por_limiar` | Calcula o F2 de cada limiar |
| `fbeta_score(y, y_pred, beta=2)` | F-beta pronto do scikit-learn, usado na tabela |
| `confusion_matrix(...).ravel()` | TN, FP, FN, TP |
| `y_scores_sgd`, `y_scores_forest` | Pontuações fora da amostra (reutilizadas na Etapa 7) |
| `limiar_sgd`, `limiar_forest` | **Limiares escolhidos, que serão aplicados sem alteração no teste (Etapa 8)** |
| `df_limiares` | Tabela padrão × ajustado |

---

## 4. Conceitos e fórmulas

**Regra de decisão com limiar $t$:**

$$\hat{y} = \begin{cases} 1 \text{ (Maligno)} & \text{se } s(\mathbf{x}) \geq t \\ 0 \text{ (Benigno)} & \text{caso contrário} \end{cases}$$

- **Baixar $t$:** mais tumores passam a ser classificados como malignos, então FN ↓, recall ↑, FP ↑ e precisão ↓.
- **Subir $t$:** o efeito inverso.

O recall só pode subir (ou ficar igual) quando o limiar desce, porque nenhum maligno já detectado "deixa" de ser detectado. A precisão não é monótona: pode oscilar, como se vê nos degraus da curva.

**F-beta:**

$$F_\beta = (1+\beta^2)\,\frac{P \cdot R}{\beta^2 P + R} \qquad F_2 = \frac{5PR}{4P + R}$$

Com $\beta = 2$, o recall tem **peso 4** ($\beta^2$) no denominador harmônico, ou seja, "vale o dobro" da precisão. Com $\beta = 1$, volta-se ao F1. O F2 é um critério **padronizado e justificável** para "priorizar o recall" sem escolher um número arbitrário como "recall ≥ 98%".

**Por que fora da amostra:** se o limiar fosse escolhido com pontuações de dados vistos no treino, as pontuações seriam otimistas, principalmente na RF, que tende a dar probabilidade ≈ 1 aos exemplos que memorizou. O limiar não funcionaria em dados novos.

**SGD × RF nas pontuações:** a `decision_function` do SGD não é probabilidade (vai de cerca de −147 a 509), e o limiar 0 é o ponto onde ela muda de sinal. A RF dá probabilidades entre 0 e 1. Com 100 árvores, os valores são múltiplos de 0,01, o que explica os poucos limiares distintos (76).

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Limiar escolhido por **máximo F2** | Critério explícito e padrão na literatura para priorizar recall; a especificação proíbe impor meta arbitrária de recall |
| β = 2 (e não 3, 5...) | É o valor convencional para "recall mais importante que precisão" sem ignorar a precisão. Um β maior levaria o limiar para recall ainda maior, com muito mais FP |
| Pontuações de `cross_val_predict` com o mesmo `cv` | Previsões fora da amostra e mesmas divisões da Etapa 5, então os números são comparáveis (no limiar padrão, os FN coincidem: 7 e 9) |
| Não usar o teste | Escolher o limiar no teste tornaria a avaliação final otimista; a especificação proíbe |
| Dummy fora da análise | Ele não tem pontuação que varie entre os tumores; qualquer limiar dá "sempre benigno" ou "sempre maligno" |
| Gráficos com zoom | Sem zoom, as diferenças relevantes (poucos pontos percentuais) ficavam invisíveis |
| Uma pequena função (`f2_por_limiar`) | Evita repetir o mesmo cálculo para os dois modelos, uma exceção prevista na especificação |
| Não usar `TunedThresholdClassifierCV` | O scikit-learn tem uma classe que faz isso automaticamente, mas o cálculo explícito é mais didático e segue o estilo da referência (`precision_recall_curve`) |

---

## 6. Interpretação dos resultados obtidos

**Pontuações e limiares:**

- SGD: pontuações de −147,40 a 508,64, com 455 limiares avaliados. **Limiar F2 = −2,9289** (padrão: 0).
- Random Forest: probabilidades de 0,00 a 1,00, com 76 limiares avaliados. **Limiar F2 = 0,45** (padrão: 0,5).

**Comparação no treino (fora da amostra, 170 malignos e 285 benignos):**

| Modelo - limiar | Precisão | Recall | F1 | F2 | FN | FP |
|---|---|---|---|---|---|---|
| SGD - padrão (0) | 95,88% | 95,88% | 95,88% | 95,88% | 7 | 7 |
| SGD - F2 (−2,929) | 92,74% | **97,65%** | 95,13% | 96,62% | **4** | 13 |
| Random Forest - padrão (0,5) | 94,15% | 94,71% | 94,43% | 94,59% | 9 | 10 |
| Random Forest - F2 (0,45) | 92,66% | **96,47%** | 94,52% | 95,68% | **6** | 13 |

**Leitura:**

1. **Consistência com a Etapa 5:** no limiar padrão, os FN (7 e 9) e os recalls (95,88% e 94,71%) são os mesmos da validação cruzada. A precisão da RF difere levemente (94,15% contra 94,18%) porque aqui as previsões das 5 dobras são **agrupadas** antes do cálculo, enquanto na Etapa 5 era feita a **média** das 5 precisões.
2. **SGD:** o novo limiar detectou **3 malignos a mais** (FN 7 → 4) e gerou **6 alarmes falsos a mais** (FP 7 → 13). Recall +1,77 ponto e precisão −3,14 pontos.
3. **Random Forest:** também **3 malignos a mais** (FN 9 → 6), com **3 alarmes falsos a mais** (FP 10 → 13). Recall +1,76 ponto e precisão −1,49 ponto.
4. **O *trade-off* aparece nos dois:** o recall sobe e a precisão cai. O F1 quase não muda (é simétrico); o F2 aumenta, como esperado do critério.
5. **Os gráficos:** no zoom da curva P×R, as estrelas (F2) ficam à direita e abaixo dos círculos (padrão), com mais recall e menos precisão. Nas curvas × limiar, a linha F2 fica à esquerda da padrão, onde o recall é mais alto e a precisão mais baixa. Em boa parte das faixas de limiar, a curva do SGD está acima da curva da RF, mas elas se cruzam em alguns trechos.
6. **Sensibilidade do F2 perto do máximo:** em uma verificação auxiliar (fora do notebook), os limiares vizinhos do SGD têm F2 entre 0,9615 e 0,9662, e os da RF, entre 0,9496 e 0,9568. O máximo não é um pico isolado, mas a escolha exata depende de poucos tumores.

**Custo clínico (no contexto acadêmico):** em 455 tumores do treino, mudar o limiar significou encontrar 3 cânceres que antes passariam despercebidos, em troca de 3 (RF) ou 6 (SGD) pacientes com tumor benigno encaminhados a exames complementares. Para triagem, essa troca é coerente com o objetivo do exercício. **O efeito real em dados novos só será conhecido no teste (Etapa 8).**

---

## 7. Erros comuns e limitações

- **Escolher o limiar no teste:** é o erro mais comum, e torna o resultado final otimista. Aqui foi evitado.
- **Usar pontuações "dentro da amostra"** (`sgd_clf.decision_function(X_train)` com o modelo treinado no treino todo): as pontuações seriam otimistas e o limiar enviesado.
- **Desalinhar os arrays:** `thresholds` tem um elemento a menos que `precisions` e `recalls`, daí o `[:-1]`.
- **Confundir `>` com `>=`:** o `predict` usa `>`, e o `precision_recall_curve` define os limiares com `>=`. Em pontuações contínuas, a diferença quase nunca importa, mas foi mantida a convenção correta em cada caso.
- **Limitação 1, limiar "colado" no treino:** o limiar escolhido é exatamente a pontuação de um tumor do treino. No SGD, a pontuação imediatamente abaixo é −3,0412, uma folga pequena. Em dados novos, a precisão e o recall exatos podem variar.
- **Limitação 2, pontuações de modelos diferentes:** no `cross_val_predict`, as pontuações de cada dobra vêm de um modelo diferente. No SGD, a escala da `decision_function` pode variar um pouco entre esses modelos. É a mesma abordagem da referência e é aceitável, mas não é perfeita.
- **Limitação 3, poucos dados:** as mudanças observadas são de 3 FN e de 3 a 6 FP; poucos tumores mudam os percentuais de forma visível.
- **Não há "limiar ótimo universal":** o F2 reflete uma preferência (recall 2× mais importante). Outra preferência daria outro limiar.

---

## 8. Perguntas que o professor poderia fazer

1. **O que é o limiar de decisão?** O valor de corte da pontuação a partir do qual o modelo classifica como maligno: 0 na `decision_function` do SGD e 0,5 na probabilidade da RF.
2. **Por que baixar o limiar aumenta o recall?** Mais tumores passam a ser classificados como malignos, então menos malignos ficam de fora (menos FN).
3. **Qual o custo?** Mais benignos também são classificados como malignos (mais FP), e a precisão cai: aqui, +6 FP no SGD e +3 FP na RF.
4. **Por que F2?** É um critério padrão que dá ao recall o dobro do peso da precisão. Assim, a prioridade do exercício é atendida sem uma meta arbitrária.
5. **Por que não escolher o limiar no teste?** Porque o teste seria usado para uma decisão de modelagem, e a avaliação final deixaria de ser imparcial.
6. **Por que usar `cross_val_predict` e não o modelo já treinado?** Para que cada pontuação venha de um modelo que não viu aquele tumor; senão, as pontuações seriam otimistas.
7. **Por que o Dummy não aparece?** Ele não gera pontuações que diferenciem os tumores; mudar o limiar não o faz aprender nada.
8. **Por que a RF só tem 76 limiares?** Suas probabilidades são médias de 100 árvores, ou seja, múltiplos de 0,01, e há poucos valores distintos.

---

## 9. Como apresentar oralmente

> "Os modelos decidem comparando uma pontuação com um limiar: 0 no SGD e 0,5 de probabilidade na Random Forest. Mudar esse limiar troca falsos negativos por falsos positivos. Para escolher um limiar sem usar o teste, gerei pontuações fora da amostra com `cross_val_predict`, nas mesmas 5 dobras da validação cruzada, e calculei precisão e recall para cada limiar possível. Como o objetivo é priorizar o recall sem impor uma meta arbitrária, escolhi o limiar que maximiza o F2, que dá ao recall o dobro do peso da precisão. No SGD o limiar foi para −2,93 e na Random Forest para 0,45. Nos dois casos, detectamos 3 cânceres a mais no treino; o custo foi de 6 alarmes falsos a mais no SGD e 3 na Random Forest. A curva e a tabela mostram esse *trade-off*. Esses limiares ficam congelados e serão aplicados no teste na etapa final."

Mostrar na tela: o zoom da curva P×R (círculos × estrelas), as curvas × limiar e a tabela FN/FP.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Previsões fora da amostra e pontuações**

| Item | Para conferir | Link |
|---|---|---|
| `cross_val_predict` | Parâmetro `method` (`decision_function`, `predict_proba`) e o aviso sobre usar essas previsões como métrica | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |
| Guia de previsões por validação cruzada | Quando e como usar | [scikit-learn — Obtaining predictions by cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#obtaining-predictions-by-cross-validation) |
| `SGDClassifier.decision_function` | Pontuação do SGD | [scikit-learn — SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) |
| `RandomForestClassifier.predict_proba` | Como a probabilidade é calculada (média das árvores) | [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |

**Curva precisão-recall e limiar**

| Item | Para conferir | Link |
|---|---|---|
| `precision_recall_curve` | Formato do retorno (n+1 precisões/recalls e n limiares) | [scikit-learn — precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) |
| Exemplo oficial | Curvas precisão-recall comentadas | [scikit-learn — Precision-Recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) |
| Ajuste do limiar de decisão | Guia oficial sobre limiar em classificação | [scikit-learn — Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html) |
| `TunedThresholdClassifierCV` | Alternativa automática, citada e não usada | [scikit-learn — TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) |

**Métricas**

| Item | Para conferir | Link |
|---|---|---|
| `fbeta_score` | Parâmetro `beta` e fórmula | [scikit-learn — fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) |
| Precisão, recall e F-measures | Fórmulas no guia do usuário | [scikit-learn — Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) |
| `confusion_matrix` | Ordem dos elementos (TN, FP, FN, TP com `.ravel()`) | [scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |

**NumPy e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `np.divide` com `where`/`out` | Divisão segura (sem divisão por zero) | [NumPy — numpy.divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) |
| `argmax` | Posição do maior valor (primeira, em caso de empate) | [NumPy — numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) |
| `ax.plot` | Curvas e marcadores (`'o'`, `'*'`) | [Matplotlib — Axes.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) |
| `ax.axvline` | Linhas verticais dos limiares | [Matplotlib — Axes.axvline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html) |
