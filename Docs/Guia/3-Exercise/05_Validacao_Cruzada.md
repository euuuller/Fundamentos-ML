# Etapa 5 — Avaliação com validação cruzada e comparação das métricas macro

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seção **4. Avaliação de Performance** (métricas por classe e formas de média), **4.1 Validação Cruzada** e **4.2 Gráfico Resumo**.
Correspondência na referência: seção 4 de `Docs/reference/2-Arquivo.ipynb` (`cross_validate` com `accuracy`, `precision_macro`, `recall_macro`, `f1_macro` por modelo) e 4.5 (gráfico-resumo). Adaptações: `StratifiedKFold(5, shuffle=True)` único em vez de `cv=3` (A2); modelos com padronização em Pipeline (A1); `zero_division=0` (A6). As matrizes de confusão, que na referência ficam junto de cada modelo na seção 4, vêm na Etapa 6, agora fora da amostra.

---

## 1. Objetivo e relação com o exercício

O enunciado pede **acurácia, precisão, recall e F1**. Em multiclasse, as três últimas são calculadas por classe e resumidas por uma **média**. Esta etapa:

1. explica as métricas por classe e as três formas de média: **macro, ponderada e micro**;
2. estima as métricas dos 4 modelos por **validação cruzada estratificada de 5 dobras**, só com o treino;
3. mostra a variação entre as dobras e compara as formas de média do F1;
4. resume tudo num gráfico, no estilo da referência.

---

## 2. Código implementado, por blocos

### Bloco 1 — Métricas por classe e formas de média (Markdown)

Fórmulas de precisão, recall e F1 da classe $k$ (tratada como "positiva" contra as demais), tabela macro × ponderada × micro e fórmulas das médias.

### Bloco 2 — Configuração

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metricas = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': 'recall_macro',
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'f1_micro': 'f1_micro',
}
```

- As quatro primeiras são as métricas da referência. `precision_macro` e `f1_macro` usam `make_scorer` para passar `zero_division=0`.
- `f1_weighted` e `f1_micro` entram **só para comparar** as formas de média; não aparecem no gráfico.

### Bloco 3 — Validação cruzada de cada modelo

```python
resultados_sgd = cross_validate(sgd_clf, X_train, y_train, cv=cv, scoring=metricas)
for chave, nome in nomes_metricas.items():
    print(f"{nome:18s} {resultados_sgd['test_' + chave].mean():.2%} ± {resultados_sgd['test_' + chave].std():.2%}")
```

Repetido para `dummy_clf`, `forest_clf` e `svc_clf`, uma célula por modelo, como na referência. Todos recebem **`X_train` bruto**, e os Pipelines do SGD e do SVC padronizam dentro de cada dobra.

### Bloco 4 — F1 macro por dobra

Tabela 5 dobras × 4 modelos, para ver a variação por trás das médias.

### Bloco 5 — Comparação das formas de média

Tabela com F1 macro, F1 ponderado, F1 micro e acurácia de cada modelo.

### Bloco 6 — Tabela-resumo e gráfico (4.2)

Médias das 4 métricas principais e gráfico de barras no estilo da seção 4.5 da referência (`viridis`, valores sobre as barras).

### Bloco 7 — Observações (Markdown)

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `StratifiedKFold(5, shuffle=True, random_state=42)` | 5 dobras de ≈28 amostras, com as proporções das classes, embaralhadas (dados ordenados) e reproduzíveis |
| `cross_validate(..., scoring=metricas)` | Clona, treina e avalia o modelo em cada dobra |
| `average='macro'` / `'weighted'` / `'micro'` | Forma de resumir as métricas por classe |
| `'recall_macro'`, `'f1_micro'` | Avaliadores prontos do scikit-learn |
| `make_scorer(..., zero_division=0)` | Precisão e F1 = 0 para classes nunca previstas (Dummy) |
| `cv`, `metricas`, `nomes_metricas` | Reutilizados nas próximas etapas |
| `resultados_sgd`, `resultados_dummy`, `resultados_forest`, `resultados_svc` | Resultados por dobra |
| `resultados_cv` | Dicionário modelo → resultados |
| `f1_por_dobra`, `df_comparacao` | Tabelas por dobra e resumo |

---

## 4. Conceitos e fórmulas

**Por classe** (classe $k$ como positiva, as demais como negativas):

$$Precision_k = \frac{TP_k}{TP_k + FP_k} \qquad Recall_k = \frac{TP_k}{TP_k + FN_k} \qquad F1_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

**Formas de média** ($N$ = 3 classes; $n_k$ = amostras da classe $k$; $n$ = total):

$$M_{macro} = \frac{1}{N}\sum_k M_k \qquad M_{ponderada} = \sum_k \frac{n_k}{n} M_k \qquad M_{micro} = M\Big(\sum_k TP_k, \sum_k FP_k, \sum_k FN_k\Big)$$

- **Macro:** cada cultivar pesa igual, então uma classe pequena mal classificada derruba a média. É a mais exigente e a usada na referência.
- **Ponderada:** classes maiores pesam mais.
- **Micro:** soma os acertos e erros de todas as classes. Com um rótulo por amostra, todo FP de uma classe é um FN de outra, então micro-precisão = micro-recall = micro-F1 = **acurácia**.

**Por que o Dummy tem recall macro de exatamente 1/3:** ele prevê sempre `class_1`, então o recall é 100% na `class_1` e 0% nas outras: $(1 + 0 + 0)/3 = 33{,}33\%$. A precisão macro é $(\text{proporção de class\_1} + 0 + 0)/3 \approx 40{,}12\%/3 = 13{,}37\%$.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| 5 dobras estratificadas e embaralhadas (referência: `cv=3`) | Dados ordenados por classe; 142 amostras (5 dobras dão mais avaliações e treinos maiores); mesmo padrão dos exercícios anteriores |
| Um único `cv` | Os 4 modelos são comparados nas mesmas divisões |
| Pipelines no SGD e no SVC | Padronização reajustada em cada dobra, sem vazamento (na referência, o *scaler* era ajustado antes) |
| Média macro como principal | Igual à referência; dá peso igual às 3 cultivares |
| F1 ponderado e micro só como comparação | Torna concreta a diferença entre as médias, exigida pela especificação |
| `zero_division=0` | O Dummy nunca prevê duas classes; evita avisos e mantém o valor 0 |
| Desvio padrão e tabela por dobra | Mostram a incerteza, essencial num conjunto pequeno |
| Matrizes de confusão só na Etapa 6 | Na referência ficam junto de cada modelo; aqui foram separadas por etapa (plano) e serão feitas fora da amostra |

---

## 6. Interpretação dos resultados obtidos

**Média ± desvio padrão (5 dobras):**

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Dummy | 40,12% ± 1,03 | 13,37% ± 0,34 | 33,33% ± 0,00 | 19,09% ± 0,35 |
| SGD | **99,29%** ± 1,43 | **99,26%** ± 1,48 | **99,39%** ± 1,21 | **99,29%** ± 1,42 |
| Random Forest | 97,91% ± 2,77 | 97,85% ± 2,93 | 98,28% ± 2,25 | 97,89% ± 2,85 |
| SVC (RBF) | 98,62% ± 1,69 | 98,65% ± 1,72 | 98,78% ± 1,51 | 98,65% ± 1,67 |

**F1 macro por dobra:**

| Dobra | Dummy | SGD | Random Forest | SVC |
|---|---|---|---|---|
| 1 | 19,51% | 100,00% | 92,80% | 96,33% |
| 2 | 19,51% | 100,00% | 100,00% | 96,91% |
| 3 | 18,80% | 100,00% | 100,00% | 100,00% |
| 4 | 18,80% | 100,00% | 100,00% | 100,00% |
| 5 | 18,80% | 96,45% | 96,66% | 100,00% |

**Formas de média do F1:**

| Modelo | F1 macro | F1 ponderado | F1 micro | Acurácia |
|---|---|---|---|---|
| Dummy | 19,09% | 22,99% | 40,12% | 40,12% |
| SGD | 99,29% | 99,29% | 99,29% | 99,29% |
| Random Forest | 97,89% | 97,93% | 97,91% | 97,91% |
| SVC | 98,65% | 98,62% | 98,62% | 98,62% |

**Leitura:**

1. **Dummy:** a acurácia de 40,12% é a proporção de `class_1` no treino; o recall macro é exatamente 1/3, como explicado acima. É a linha de base.
2. **Modelos reais com desempenho muito alto:** F1 macro entre 97,89% e 99,29%. As cultivares do Wine são bem separáveis pelos atributos químicos, como a Etapa 2 sugeria.
3. **Diferenças pequenas e não conclusivas:** a distância entre o melhor (SGD, 99,29%) e o pior (RF, 97,89%) é de 1,4 ponto, ou seja, cerca de 2 amostras em 142. Os desvios entre dobras (1,4 a 2,9 pontos) são **maiores** que essas diferenças, e na tabela por dobra **cada modelo tem dobras com 100% e dobras com erro**. Não há modelo claramente superior entre SGD, SVC e RF.
4. **Médias macro × ponderada × micro:** nos modelos reais, as três praticamente coincidem, porque todas as classes são bem classificadas. No Dummy, divergem muito (19,09% × 22,99% × 40,12%): a **macro** é a mais severa com um modelo que ignora classes, e a **micro** é igual à acurácia, como manda a teoria.
5. **Comparação com a referência (MNIST):** lá, os modelos ficaram entre 90% (SGD) e 96,5% (RF), e o SGD era claramente o pior. No Wine, com 3 classes bem separadas, o SGD linear fica no mesmo nível dos outros.

---

## 7. Erros comuns e limitações

- **Usar `cv=3` sem `shuffle` em dados ordenados:** sem estratificação, as dobras poderiam ter só uma cultivar. O `StratifiedKFold` explícito com `shuffle` evita qualquer dependência da ordem.
- **Padronizar antes do `cross_validate`:** vazamento (erro da referência).
- **Ler 99% × 98% como diferença real:** são 1 ou 2 amostras, dentro da variação entre dobras.
- **Confundir micro com macro:** a micro é a acurácia; a macro dá peso igual às classes.
- **Esquecer `zero_division` no Dummy:** gera avisos e pode confundir a leitura.
- **Limitação:** 142 amostras; cada dobra tem ≈28. Uma amostra a mais ou a menos muda a métrica da dobra em ≈3,5 pontos.

---

## 8. Perguntas que o professor poderia fazer

1. **O que é a média macro?** A média simples das métricas de cada classe; cada cultivar pesa igual.
2. **Qual a diferença para a ponderada e a micro?** A ponderada pesa cada classe pelo seu tamanho; a micro soma os acertos e erros de todas as classes e, aqui, é igual à acurácia.
3. **Por que o recall macro do Dummy é 33,33%?** Ele acerta 100% de uma classe e 0% das outras duas: média 1/3.
4. **Qual modelo foi melhor?** O SGD teve a maior média (F1 macro 99,29%), mas a diferença para SVC e RF é de 1 ou 2 amostras, menor que a variação entre dobras; não há superioridade clara.
5. **Por que 5 dobras com `shuffle`?** Os dados vêm ordenados por classe e o treino é pequeno; o embaralhamento garante dobras representativas.
6. **O desempenho alto é suspeito?** Não necessariamente: o Wine é um dataset conhecido por ter classes bem separáveis, e a avaliação é fora da amostra e sem vazamento.

---

## 9. Como apresentar oralmente

> "Em multiclasse, precisão, recall e F1 são calculados para cada cultivar e depois resumidos. Usei a média macro, como na referência, que dá o mesmo peso às três cultivares. Avaliei os quatro modelos com validação cruzada estratificada de 5 dobras, embaralhada porque o dataset vem ordenado, e com a padronização dentro do pipeline. O Dummy acertou 40%, a proporção da classe mais comum, com recall macro de um terço, porque só acerta uma classe. Os três modelos reais ficaram acima de 97%: SGD com 99,3%, SVC com 98,6% e Random Forest com 97,9% de F1 macro. Mas a diferença entre eles é de uma ou duas amostras, e cada modelo teve dobras com 100% e dobras com erro, então não aponto um vencedor. A tabela das médias mostra que a macro é a mais exigente: no Dummy, ela dá 19%, enquanto a micro, que é igual à acurácia, dá 40%."

Mostrar na tela: a tabela macro × ponderada × micro, o gráfico-resumo e a tabela de F1 por dobra.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Métricas multiclasse**

| Item | Para conferir | Link |
|---|---|---|
| Do binário ao multiclasse | Macro, ponderada (*weighted*) e micro | [scikit-learn — From binary to multiclass and multilabel](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel) |
| Métricas de classificação multiclasse | Visão geral | [scikit-learn — Multiclass and multilabel classification](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification) |
| `precision_score` | `average`, `zero_division` | [scikit-learn — precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) |
| `recall_score` | `average` | [scikit-learn — recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) |
| `f1_score` | `average`, `zero_division` | [scikit-learn — f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| Parâmetro `scoring` | Nomes prontos (`'recall_macro'`, `'f1_micro'`...) | [scikit-learn — The scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) |
| `make_scorer` | Avaliadores com parâmetros extras | [scikit-learn — make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) |

**Validação cruzada**

| Item | Para conferir | Link |
|---|---|---|
| `StratifiedKFold` | `shuffle`, `random_state` | [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) |
| `cross_validate` | Retorno `test_<métrica>` | [scikit-learn — cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) |
| Múltiplas métricas | `scoring` com dicionário | [scikit-learn — Multiple metric evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.plot.bar` | Barras agrupadas | [pandas — DataFrame.plot.bar](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html) |
| `ax.bar_label` | Rótulos das barras | [Matplotlib — Axes.bar_label](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html) |
