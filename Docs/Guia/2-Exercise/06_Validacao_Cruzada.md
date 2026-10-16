# Etapa 6 — Avaliação com validação cruzada e comparação das métricas (com AUC)

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **4. Avaliação de Performance** (definições, incluindo curva ROC e AUC) e **4.1 Validação Cruzada**.
Correspondência na referência: seção 4 e 4.1 de `Docs/reference/1-Arquivo.ipynb` (`cross_validate` por modelo + gráfico comparativo). Mesma estrutura da Etapa 5 do Exercício 01, com a **AUC** acrescentada (exigida pelo enunciado do Exercício 2).

---

## 1. Objetivo e relação com o exercício

O enunciado pede **acurácia, precisão, recall, F1-score, matriz de confusão, curva ROC e AUC**. Esta etapa:

1. define as métricas no contexto do Titanic (o que é FN e FP aqui) e apresenta a **curva ROC** e a **AUC**;
2. estima as cinco métricas numéricas de cada modelo por **validação cruzada estratificada de 5 dobras**, só com o treino;
3. compara os modelos em tabela (média ± desvio e por dobra) e em gráfico.

As curvas ROC propriamente ditas são desenhadas na Etapa 7, e a matriz de confusão fica para a Etapa 8.

---

## 2. Código implementado, por blocos

### Bloco 1 — Definições (Markdown)

Tabela TP/TN/FP/FN no contexto do Titanic, as fórmulas de acurácia, precisão, recall e F1 e a explicação de **TPR, FPR, curva ROC e AUC**. Destaca que, ao contrário do Exercício 01, **FN e FP têm importância equivalente**, e por isso o F1 é o critério de equilíbrio.

### Bloco 2 — Configuração

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metricas = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc'
}
```

Igual ao Exercício 01, com `'roc_auc'` acrescentado. Esse avaliador pronto usa a **pontuação contínua** de cada modelo, e não a classe prevista: `decision_function` no SGD e `predict_proba` na RF e no Dummy.

### Bloco 3 — Validação cruzada de cada modelo

```python
resultados_sgd = cross_validate(sgd_clf, X_train, y_train, cv=cv, scoring=metricas)
for chave, nome in nomes_metricas.items():
    print(f"{nome:9s} {resultados_sgd['test_' + chave].mean():.2%} ± {resultados_sgd['test_' + chave].std():.2%}")
```

Repetido para `dummy_clf` e `forest_clf`. Em cada dobra, o `cross_validate` **clona o Pipeline inteiro** e o treina nas 4 dobras restantes. Isso inclui o pré-processamento: medianas, moda e padronização são reaprendidas sem ver a dobra de validação.

### Bloco 4 — F1 e AUC por dobra

Tabela com F1 e AUC do SGD e da RF em cada uma das 5 dobras, para ver a **consistência** da comparação.

### Bloco 5 — Tabela-resumo e gráfico

Médias das 5 métricas por modelo e gráfico de barras no estilo da referência (`viridis`, rótulos sobre as barras, aqui na vertical porque são 15 barras).

### Bloco 6 — Observações (Markdown)

Interpretação no notebook.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `StratifiedKFold(5, shuffle=True, random_state=42)` | 5 dobras com a mesma proporção de sobreviventes, reproduzíveis |
| `cross_validate(..., scoring=metricas)` | Clona, treina e avalia o Pipeline em cada dobra; retorna `test_<métrica>` por dobra |
| `'roc_auc'` | Avaliador de AUC; usa pontuações contínuas (`decision_function`/`predict_proba`) |
| `make_scorer(..., zero_division=0)` | Precisão e F1 = 0 quando não há previsões positivas (Dummy) |
| `cv`, `metricas`, `nomes_metricas` | Reutilizados nas próximas etapas |
| `resultados_sgd`, `resultados_dummy`, `resultados_forest` | Resultados por dobra |
| `por_dobra` | F1 e AUC dos modelos reais em cada dobra |
| `df_comparacao` | Médias por modelo e métrica (base do gráfico) |

---

## 4. Conceitos e fórmulas

**Métricas dependentes do limiar:**

$$Acc = \frac{TP+TN}{TP+TN+FP+FN} \quad Precision = \frac{TP}{TP+FP} \quad Recall = \frac{TP}{TP+FN} \quad F1 = \frac{2TP}{2TP+FP+FN}$$

**Curva ROC:** para cada limiar $t$ sobre a pontuação, calculam-se

$$TPR(t) = \frac{TP}{TP+FN} \qquad FPR(t) = \frac{FP}{FP+TN}$$

e marca-se o ponto $(FPR, TPR)$. Um limiar muito alto dá $(0, 0)$, porque ninguém é previsto como sobrevivente; um limiar muito baixo dá $(1, 1)$, porque todos são. A diagonal $TPR = FPR$ corresponde a um classificador aleatório.

**AUC:** a área sob a curva ROC. Interpretação probabilística:

$$AUC = P\big(s(\text{sobrevivente}) > s(\text{não sobrevivente})\big)$$

Ou seja, a chance de um sobrevivente sorteado receber pontuação maior que um não sobrevivente sorteado. Vale 0,5 sem poder de separação e 1 com separação perfeita.

**Por que a AUC complementa as outras métricas:** ela avalia a **ordenação** das pontuações, independentemente do limiar. Dois modelos podem ter o mesmo F1 num limiar e AUCs diferentes. A AUC diz qual ordena melhor os passageiros, e o F1 diz qual decide melhor naquele limiar.

**Validação cruzada:** 712 passageiros em 5 dobras de 142 ou 143, cada uma com ≈54–55 sobreviventes. Cada passageiro é avaliado uma vez, por um modelo que não o viu.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Mesmo `cv` do Exercício 01 (5 dobras, estratificado, embaralhado) | Treino pequeno; comparação justa (mesmas divisões para todos os modelos) |
| Acrescentar `roc_auc` | Exigência do enunciado; mede a separação independentemente do limiar |
| Pré-processamento dentro do Pipeline | Reajustado a cada dobra, sem vazamento (exigência da especificação) |
| `zero_division=0` | O Dummy nunca prevê Sobreviveu; evita avisos e mantém o valor 0 |
| Tabela por dobra com F1 e AUC | O F1 é o critério de equilíbrio adotado (o limiar será escolhido por ele) e a AUC é a métrica nova; mostrar as dobras revela a consistência da comparação |
| Avaliar só no treino | O teste fica reservado para a Etapa 9 |

---

## 6. Interpretação dos resultados obtidos

**Médias ± desvio padrão (5 dobras, limiar padrão):**

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Dummy | 61,66% ± 0,27% | 0,00% | 0,00% | 0,00% | 50,00% |
| SGD | 71,90% ± 4,40% | 62,78% ± 6,81% | 69,58% ± **10,40%** | 65,33% ± 5,34% | 78,75% ± 3,76% |
| Random Forest | **78,51%** ± 1,45% | **71,03%** ± 3,16% | **74,72%** ± 1,45% | **72,76%** ± 0,98% | **86,31%** ± 0,75% |

**F1 e AUC por dobra:**

| Dobra | SGD - F1 | SGD - AUC | RF - F1 | RF - AUC |
|---|---|---|---|---|
| 1 | 69,64% | 81,37% | 73,21% | 85,30% |
| 2 | 65,35% | 72,43% | 72,57% | 85,57% |
| 3 | 71,43% | 83,08% | 72,41% | 86,58% |
| 4 | 64,18% | 80,00% | 71,30% | 86,82% |
| 5 | 56,07% | 76,85% | 74,29% | 87,27% |

**Leitura:**

1. **Dummy:** a acurácia de 61,66% é a proporção de não sobreviventes no treino; recall, precisão e F1 são 0 e a **AUC é exatamente 50%**, porque ele dá a mesma probabilidade a todos os passageiros e não ordena ninguém. O pequeno desvio na acurácia (0,27) vem de as dobras terem 142 ou 143 passageiros.
2. **Random Forest:** é a melhor em **todas** as métricas, com AUC de 86,31% e **baixa variação** entre as dobras (desvio de 0,75 ponto na AUC e de 0,98 no F1).
3. **SGD:** desempenho menor e **instável**. O recall varia ±10,40 pontos, e o F1 vai de 56,07% (dobra 5) a 71,43% (dobra 3). Ele é sensível à divisão dos dados com os hiperparâmetros padrão da referência.
4. **Consistência:** a RF teve F1 e AUC maiores que o SGD **nas 5 dobras**. Diferentemente do Exercício 01, onde a diferença entre os modelos ficava dentro da variação, aqui a vantagem da RF é consistente entre as divisões.
5. **Dificuldade do problema:** os valores (F1 ≈ 65–73%) são bem menores que no câncer (≈95%). Com 7 atributos simples, a sobrevivência é menos previsível. Os modelos reais, ainda assim, superam claramente o Dummy.
6. **AUC × F1:** a AUC da RF (86%) indica uma boa ordenação dos passageiros; o F1 (73%) mostra a qualidade da decisão no limiar padrão. A Etapa 7 investiga se outro limiar melhora o F1.

---

## 7. Erros comuns e limitações

- **Usar a classe prevista (0/1) para calcular a AUC:** a AUC precisa da pontuação contínua. O avaliador `'roc_auc'` cuida disso.
- **Achar que AUC alta garante boa acurácia:** a AUC mede ordenação; a acurácia depende do limiar.
- **Pré-processar fora do Pipeline antes do `cross_validate`:** seria vazamento.
- **Ler o desvio como intervalo de confiança:** são só 5 valores, com dobras dependentes.
- **Limitação:** o SGD usa hiperparâmetros padrão; sua instabilidade pode refletir essa configuração, e não um limite do modelo linear em si. Ajustar hiperparâmetros está fora do escopo.
- **Limitação:** todas as métricas exceto a AUC usam o limiar padrão.

---

## 8. Perguntas que o professor poderia fazer

1. **O que a AUC mede?** A probabilidade de o modelo dar uma pontuação maior a um sobrevivente do que a um não sobrevivente; é a área sob a curva ROC.
2. **Por que a AUC do Dummy é 50%?** Ele dá a mesma pontuação a todos, então não ordena ninguém; equivale a sortear.
3. **Qual modelo é melhor?** A Random Forest, em todas as métricas e nas 5 dobras (F1 e AUC), com variação pequena.
4. **Por que o SGD varia tanto?** Com os hiperparâmetros padrão, o resultado depende bastante da divisão dos dados: o recall varia ±10 pontos e o F1 vai de 56% a 71%.
5. **O pré-processamento foi feito em cada dobra?** Sim: o Pipeline inteiro é clonado e reajustado só com as 4 dobras de treino de cada rodada.
6. **Por que o desempenho é menor que no Exercício 01?** O problema é mais difícil: com 7 atributos simples, os padrões de sobrevivência se sobrepõem mais que os padrões dos tumores.
7. **Qual a diferença entre AUC e F1?** A AUC independe do limiar e avalia a ordenação; o F1 avalia a decisão num limiar específico.

---

## 9. Como apresentar oralmente

> "Avaliei os três modelos com validação cruzada estratificada de 5 dobras, só no treino. Como o pré-processamento está no pipeline, ele foi refeito em cada dobra, sem vazamento. Além das quatro métricas do primeiro exercício, incluí a AUC, a área sob a curva ROC. Ela mede se o modelo dá pontuações maiores aos sobreviventes do que aos não sobreviventes, independentemente do limiar: 50% é sortear e 100% é separar perfeitamente. O Dummy tem 62% de acurácia, mas AUC de 50%, o que mostra que não separa nada. A Random Forest foi a melhor em todas as métricas: F1 de 73% e AUC de 86%, com pouca variação. O SGD ficou com F1 de 65% e AUC de 79%, mas variou muito entre as dobras. A Random Forest venceu o SGD nas cinco dobras, então aqui a vantagem é consistente, diferente do primeiro exercício."

Mostrar na tela: a definição de ROC/AUC, o gráfico comparativo e a tabela por dobra.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Validação cruzada**

| Item | Para conferir | Link |
|---|---|---|
| `StratifiedKFold` | Parâmetros e divisões | [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) |
| `cross_validate` | Retorno `test_<métrica>` | [scikit-learn — cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) |
| Múltiplas métricas | `scoring` com dicionário | [scikit-learn — Multiple metric evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation) |

**Métricas**

| Item | Para conferir | Link |
|---|---|---|
| Parâmetro `scoring` | Nomes prontos (`'roc_auc'`, `'recall'`...) | [scikit-learn — The scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) |
| `make_scorer` | Avaliadores com `zero_division` | [scikit-learn — make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) |
| Precisão, recall e F | Fórmulas | [scikit-learn — Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) |
| Curva ROC (guia) | TPR, FPR, AUC | [scikit-learn — Receiver operating characteristic (ROC)](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) |
| `roc_auc_score` | Cálculo da AUC | [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) |
| `roc_curve` | Pontos da curva (usada na Etapa 7) | [scikit-learn — roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) |
| `DummyClassifier` | Por que a AUC é 0,5 | [scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.plot.bar` | Barras agrupadas | [pandas — DataFrame.plot.bar](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html) |
| `ax.bar_label` | Rótulos (`rotation=90`) | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
