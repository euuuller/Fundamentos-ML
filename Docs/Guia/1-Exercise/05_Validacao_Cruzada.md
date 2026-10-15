# Etapa 5 — Avaliação com validação cruzada e comparação das métricas

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **4. Avaliação de Performance** (definições das métricas) e **4.1 Validação Cruzada**.
Correspondência na referência: seção 4 (definições de acurácia, precisão, recall e F1) e 4.1 (`cross_validate` por modelo + gráfico comparativo) de `Docs/reference/1-Arquivo.ipynb`.

---

## 1. Objetivo e relação com o exercício

O enunciado pede para **avaliar acurácia, precisão, recall e F1-score** dos três modelos. Esta etapa:

1. define cada métrica no contexto do problema (o que é um FN e um FP em diagnóstico de câncer);
2. estima as quatro métricas de cada modelo por **validação cruzada estratificada de 5 dobras**, usando **só o treino**;
3. compara os modelos em tabela e gráfico.

O teste continua reservado. Estas estimativas servem para **comparar modelos** e serão a base das etapas 6 e 7.

---

## 2. Código implementado, por blocos

### Bloco 1 — Definições das métricas (Markdown)

Tabela TP/TN/FP/FN com o significado clínico de cada caso e as fórmulas de acurácia, precisão, recall e F1, adaptadas da referência. Os exemplos genéricos da referência (filtro de vídeos, fraudes) foram trocados pelo contexto do exercício.

### Bloco 2 — Configuração da validação cruzada

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metricas = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': make_scorer(f1_score, zero_division=0)
}
nomes_metricas = {'accuracy': 'Acurácia', 'precision': 'Precisão', 'recall': 'Recall', 'f1': 'F1-Score'}
```

- `cv` define **como** dividir o treino: 5 dobras, estratificadas, embaralhadas com semente fixa. O mesmo objeto será usado em todas as avaliações no treino (Etapas 5, 6 e 7).
- `metricas` é um dicionário `nome → avaliador`. Acurácia e recall usam os nomes prontos do scikit-learn; precisão e F1 usam `make_scorer` para passar `zero_division=0`.
- `nomes_metricas` só traduz os nomes na exibição.

### Bloco 3 — Validação cruzada de cada modelo

```python
resultados_sgd = cross_validate(sgd_clf, X_train, y_train, cv=cv, scoring=metricas)

for chave, nome in nomes_metricas.items():
    print(f"{nome:9s} {resultados_sgd['test_' + chave].mean():.2%} ± {resultados_sgd['test_' + chave].std():.2%}")
```

Repetido para `dummy_clf` e `forest_clf`. Diferenças em relação à referência:
- O SGD recebe `X_train` **bruto**, e o Pipeline padroniza dentro de cada dobra. A referência passava `X_train_scaled`, padronizado com todo o treino.
- Além da média, imprime o **desvio padrão** entre as dobras, para mostrar a variação da estimativa.
- As 4 linhas de `print` da referência viraram um laço sobre `nomes_metricas`. O resultado é o mesmo, com menos repetição.

### Bloco 4 — Recall por dobra

```python
recall_por_dobra = pd.DataFrame({
    'Dummy Classifier': resultados_dummy['test_recall'],
    'SGD Classifier': resultados_sgd['test_recall'],
    'Random Forest': resultados_forest['test_recall']
}, index=[f"Dobra {i}" for i in range(1, 6)])
```

Mostra o recall de cada modelo em cada dobra, ou seja, a variação "por trás" da média.

### Bloco 5 — Tabela comparativa e gráfico

```python
df_comparacao = pd.DataFrame({
    modelo: {nome: res['test_' + chave].mean() for chave, nome in nomes_metricas.items()}
    for modelo, res in resultados_cv.items()
}).T
```

Monta a tabela modelos × métricas (médias) e plota o gráfico de barras com o mesmo estilo da referência: `colormap='viridis'`, rótulos em % sobre as barras, legenda à direita.

### Bloco 6 — Observações (Markdown)

Interpretação dos resultados obtidos.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Divide o treino em 5 dobras com a mesma proporção de classes; `shuffle` embaralha antes de dividir; `random_state` torna as dobras reproduzíveis |
| `cross_validate(modelo, X, y, cv=cv, scoring=metricas)` | Para cada dobra: **clona** o modelo, treina nas outras 4 e avalia na dobra separada. Retorna um dicionário com `fit_time`, `score_time` e `test_<métrica>` (um valor por dobra) |
| `make_scorer(func, **kwargs)` | Transforma uma função de métrica num avaliador aceito por `scoring`, permitindo passar parâmetros extras (`zero_division=0`) |
| `zero_division=0` | Valor usado quando a métrica tem divisão por zero (precisão sem previsões positivas) |
| `resultados_sgd`, `resultados_dummy`, `resultados_forest` | Saídas do `cross_validate`, com os valores por dobra |
| `.mean()`, `.std()` | Média e desvio padrão (populacional, NumPy `ddof=0`) das 5 dobras |
| `recall_por_dobra` | Tabela 5 dobras × 3 modelos |
| `df_comparacao` | Tabela das médias, base do gráfico |
| `cv`, `metricas`, `nomes_metricas` | Reutilizados nas próximas etapas |

Observação: o `cross_validate` treina **cópias** dos modelos. Os modelos `sgd_clf`, `dummy_clf` e `forest_clf` treinados na Etapa 4 não são alterados.

---

## 4. Conceitos e fórmulas

**Matriz de possibilidades** (positivo = Maligno):

| | Previsto Benigno (0) | Previsto Maligno (1) |
|---|---|---|
| **Real Benigno (0)** | TN | FP (exame extra desnecessário) |
| **Real Maligno (1)** | **FN (câncer não detectado)** | TP |

**Métricas:**

$$Acc = \frac{TP + TN}{TP + TN + FP + FN} \qquad Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN} \qquad F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall} = \frac{2TP}{2TP + FP + FN}$$

- **Recall** é a métrica prioritária: o denominador é o total de malignos reais, então cada FN o reduz diretamente.
- **Precisão** mede o custo do outro lado: cada FP (benigno tratado como maligno) a reduz.
- **F1** usa a média harmônica, que é puxada pelo menor valor: com precisão 1,0 e recall 0,1, F1 ≈ 0,18, e não 0,55.

**Validação cruzada k-fold estratificada (k = 5):**

1. O treino (455 amostras) é dividido em 5 dobras de 91 amostras, cada uma com 34 malignos (170/5).
2. Para cada dobra $i$: treina-se em 4 dobras (364 amostras) e avalia-se na dobra $i$.
3. A estimativa é a média das 5 avaliações:

$$\overline{M} = \frac{1}{5}\sum_{i=1}^{5} M_i \qquad \sigma_M = \sqrt{\frac{1}{5}\sum_{i=1}^{5}(M_i - \overline{M})^2}$$

Cada amostra do treino é avaliada **exatamente uma vez**, por um modelo que não a viu.

**Granularidade do recall por dobra:** com 34 malignos por dobra, o recall só pode assumir valores múltiplos de $1/34 \approx 2{,}94$ pontos percentuais. Por exemplo, 33/34 = 97,06% e 31/34 = 91,18%.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| 5 dobras (a referência usa 3) | Com 455 amostras, cada modelo treina com 80% do treino (contra 67% com 3 dobras), e a média de 5 avaliações é mais estável |
| `StratifiedKFold` explícito com `shuffle=True` | Garante a mesma proporção de malignos em cada dobra; o embaralhamento evita dependência da ordem das amostras; com `random_state`, as dobras são reproduzíveis |
| Um único `cv` para todos os modelos | Todos são avaliados **nas mesmas divisões**, então as diferenças vêm dos modelos e não do acaso da divisão |
| `Pipeline` no SGD | A padronização é refeita em cada dobra, sem usar a dobra de validação. É o requisito antivazamento da especificação |
| `zero_division=0` via `make_scorer` | O Dummy nunca prevê "Maligno", então a precisão é 0/0. O valor 0 é o mesmo que a referência obteve, mas sem os avisos `UndefinedMetricWarning` poluindo a saída. O motivo está explicado no notebook, e o aviso não foi escondido globalmente |
| Mostrar desvio padrão e recall por dobra | Sem a variação, uma diferença de 1 ponto entre modelos poderia parecer conclusiva |
| Avaliar no treino (fora da amostra) e não no teste | A especificação exige reservar o teste para a avaliação final |

---

## 6. Interpretação dos resultados obtidos

**Médias ± desvio padrão nas 5 dobras:**

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| Dummy Classifier | 62,64% ± 0,00% | 0,00% ± 0,00% | 0,00% ± 0,00% | 0,00% ± 0,00% |
| SGD Classifier | 96,92% ± 0,82% | 96,08% ± 3,35% | **95,88% ± 3,00%** | 95,89% ± 1,06% |
| Random Forest | 95,82% ± 1,62% | 94,18% ± 2,56% | **94,71% ± 2,20%** | 94,43% ± 2,14% |

**Recall por dobra:**

| Dobra | Dummy | SGD | Random Forest |
|---|---|---|---|
| 1 | 0,00% | 91,18% (31/34) | 94,12% (32/34) |
| 2 | 0,00% | 97,06% (33/34) | 97,06% (33/34) |
| 3 | 0,00% | 97,06% (33/34) | 91,18% (31/34) |
| 4 | 0,00% | 94,12% (32/34) | 94,12% (32/34) |
| 5 | 0,00% | 100,00% (34/34) | 97,06% (33/34) |

As frações foram deduzidas dos percentuais e dos 34 malignos por dobra.

**Leitura:**

1. **Dummy, a armadilha da acurácia.** A acurácia de 62,64% é exatamente a proporção de benignos no treino (285/455), com recall 0: o modelo nunca detecta câncer. O desvio é 0 porque, com estratificação, todas as dobras têm a mesma proporção.
2. **SGD e Random Forest** superam o Dummy por ampla margem em todas as métricas, o que mostra que aprenderam padrões reais dos atributos.
3. **Falsos negativos no treino:** somando as dobras, o SGD deixou de detectar **7** dos 170 malignos (3+1+1+2+0) e a Random Forest, **9** (2+1+3+2+1). Isso bate com os recalls médios: 163/170 = 95,88% e 161/170 = 94,71%.
4. **SGD × Random Forest:** o SGD teve médias um pouco maiores nas quatro métricas, mas a diferença no recall (≈1,2 ponto, ou 2 tumores) é **menor que o desvio padrão entre as dobras** (2–3 pontos). Na dobra 1 a Random Forest foi melhor; na dobra 3, o SGD. **Com estes dados, não é possível afirmar que um modelo é claramente superior.**
5. **Precisão e recall estão equilibrados** nos dois modelos (≈94–96%), no limiar padrão. A Etapa 6 vai investigar se deslocar o limiar reduz os FN e quanto isso custa em FP.

---

## 7. Erros comuns e limitações

- **Padronizar antes do `cross_validate`** (como na referência): a dobra de validação influencia a média e o desvio usados no treino, o que é vazamento. O Pipeline resolve isso.
- **Comparar só as médias:** sem o desvio, diferenças pequenas parecem significativas.
- **Usar `cv=5` com inteiro achando que embaralha:** para classificadores, `cv=5` usa `StratifiedKFold` **sem** embaralhamento. Aqui o `cv` explícito deixa a escolha clara.
- **Ler o desvio padrão como intervalo de confiança:** são só 5 valores, e as dobras compartilham dados de treino. O desvio é uma indicação de variabilidade, não um teste estatístico.
- **Confundir estas métricas com o desempenho final:** elas vêm do treino (fora da amostra). A estimativa final será a do teste (Etapa 8).
- **Limitação:** todas as métricas usam o limiar padrão dos modelos (0 no `decision_function` do SGD; 0,5 de probabilidade na RF). Ainda não houve nenhuma otimização de recall.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que a acurácia do Dummy é 62,64%?** É a proporção de benignos no treino (285/455). Ele acerta todos os benignos e erra todos os malignos.
2. **Por que validação cruzada e não o teste?** O teste é reservado para a avaliação final. A validação cruzada estima o desempenho só com o treino, avaliando cada amostra com um modelo que não a viu.
3. **Por que 5 dobras e não 3, como na referência?** O dataset é pequeno (455 amostras de treino). Com 5 dobras, cada modelo treina com mais dados e a média de 5 avaliações é mais estável.
4. **Por que o Pipeline importa na validação cruzada?** Porque o `StandardScaler` aprende média e desvio. Sem o Pipeline, esses valores incluiriam a dobra de validação, o que é vazamento.
5. **Qual modelo é melhor?** O SGD teve médias ligeiramente maiores (recall 95,88% contra 94,71%), mas a diferença (2 tumores em 170) é menor que a variação entre dobras. Não dá para afirmar superioridade clara.
6. **O que significa o `zero_division=0`?** Quando o modelo não prevê nenhum maligno, a precisão é 0/0. O parâmetro define o resultado como 0, em vez de gerar aviso.
7. **Por que o recall da dobra 5 do SGD é exatamente 100%?** O modelo detectou os 34 malignos daquela dobra. Com 34 casos, cada FN muda o recall em ≈2,94 pontos.

---

## 9. Como apresentar oralmente

> "Primeiro defini as métricas no contexto do problema: o falso negativo é o câncer não detectado, por isso o recall é a prioridade, e o falso positivo é um exame extra, o que afeta a precisão. Depois avaliei os três modelos com validação cruzada estratificada de 5 dobras, só no treino. Adaptei a referência em três pontos: usei 5 dobras porque o dataset é pequeno, mantive o SGD no Pipeline para não haver vazamento e usei `zero_division=0` para o Dummy. O Dummy tem 62,6% de acurácia e recall zero, o que mostra como a acurácia engana. O SGD e a Random Forest ficaram acima de 94% em todas as métricas. O recall foi de 95,9% no SGD e 94,7% na Random Forest, ou seja, 7 e 9 tumores malignos perdidos em 170. Como essa diferença é menor que a variação entre as dobras, não afirmo que um é superior. Esses números usam o limiar padrão; na próxima etapa vou estudar o ajuste do limiar."

Mostrar na tela: a tabela TP/FP/FN/TN, o gráfico comparativo e a tabela de recall por dobra.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Validação cruzada**

| Item | Para conferir | Link |
|---|---|---|
| `StratifiedKFold` | Parâmetros `n_splits`, `shuffle`, `random_state` | [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) |
| `cross_validate` | Parâmetros e o formato do dicionário retornado (`test_<métrica>`) | [scikit-learn — cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) |
| Guia de validação cruzada | Como funciona o k-fold e o cálculo das métricas | [scikit-learn — Computing cross-validated metrics](https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics) |
| Múltiplas métricas | Uso de `scoring` com dicionário | [scikit-learn — The cross_validate function and multiple metric evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation) |
| k-fold estratificado | Ilustração das dobras estratificadas | [scikit-learn — Stratified k-fold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold) |

**Métricas**

| Item | Para conferir | Link |
|---|---|---|
| Parâmetro `scoring` | Nomes prontos (`'accuracy'`, `'recall'`...) | [scikit-learn — The scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) |
| `make_scorer` | Criar avaliadores com parâmetros extras | [scikit-learn — make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) |
| Precisão, recall e F | Definições e fórmulas no guia do usuário | [scikit-learn — Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures) |
| Acurácia | Definição | [scikit-learn — Accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) |
| `accuracy_score` | Função | [scikit-learn — accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) |
| `precision_score` | Parâmetros `pos_label` e `zero_division` | [scikit-learn — precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) |
| `recall_score` | Parâmetros `pos_label` e `zero_division` | [scikit-learn — recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) |
| `f1_score` | Parâmetros `pos_label` e `zero_division` | [scikit-learn — f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |

**NumPy, pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `np.std` | Desvio padrão (`ddof=0` por padrão) | [NumPy — numpy.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html) |
| `DataFrame.plot.bar` | Gráfico de barras agrupadas a partir do DataFrame | [pandas — DataFrame.plot.bar](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html) |
| `ax.bar_label` | Rótulos de valor sobre as barras | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
