# Etapa 6 — Matrizes de confusão no treino (fora da amostra) e análise por classe

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seções **4.3 Matrizes de Confusão (Treino, fora da amostra)** e **4.4 Análise por Classe**.
Correspondência na referência: as matrizes da seção 4 de `Docs/reference/2-Arquivo.ipynb` (absoluta, relativa com `normalize="true"` e erros com `sample_weight`). **Correção principal:** a referência gerava essas matrizes com `predict` no próprio treino (dentro da amostra); aqui elas usam `cross_val_predict` (adaptação A3). Também foi corrigido o título trocado da matriz do SVC (problema "d").

---

## 1. Objetivo e relação com o exercício

O enunciado inclui a **matriz de confusão** entre as medidas de desempenho. Em multiclasse, ela responde à pergunta que as médias da Etapa 5 não respondem: **quais cultivares são confundidas com quais?** Esta etapa:

1. gera, para os 4 modelos, previsões **fora da amostra** no treino;
2. desenha, para cada modelo, as **3 versões** da matriz usadas na referência;
3. calcula **precisão, recall e F1 de cada cultivar** e lista os **pares de confusão**.

O teste continua reservado.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Tabela das 3 versões da matriz e a justificativa de usar previsões fora da amostra.

### Bloco 2 — Função que desenha as 3 versões

```python
def plot_matrizes_confusao(y_real, y_pred, nome_modelo, conjunto):
    erros = (y_pred != y_real)
    versoes = [
        ("Absoluta", {}),
        ("Relativa", {'normalize': 'true', 'values_format': '.0%'}),
        ("Somente erros", {'sample_weight': erros, 'normalize': 'true', 'values_format': '.0%'}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (titulo, opcoes) in zip(axes, versoes):
        ConfusionMatrixDisplay.from_predictions(y_real, y_pred, display_labels=rotulos,
                                                cmap=plt.cm.Blues, colorbar=False, ax=ax, **opcoes)
        ...
```

- Na referência, as 3 matrizes de cada modelo eram 3 células repetidas, ou seja, 12 células para 4 modelos, com um título errado. Aqui, uma função gera as 3 versões **lado a lado, numa figura por modelo**, com nomes das classes, eixos e título com modelo e conjunto. *(No teste, Etapa 7, a decisão foi mostrar só a matriz relativa, como na referência; por isso, lá é usada uma célula simples por modelo, e não esta função.)*
- `**opcoes` repassa os parâmetros específicos de cada versão para `from_predictions`.
- `colorbar=False` deixa as três matrizes compactas lado a lado; os valores já aparecem nas células.
- O tamanho das fontes não é alterado com `plt.rc` (adaptação A7).

### Bloco 3 — Previsões fora da amostra

```python
previsoes_treino = {nome: cross_val_predict(modelo, X_train, y_train, cv=cv)
                    for nome, modelo in modelos.items()}
```

Mesmo `cv` da Etapa 5. Cada amostra é prevista pelo modelo treinado nas outras 4 dobras. Também imprime o número de erros de cada modelo.

### Bloco 4 — Uma figura por modelo

Na ordem da referência: SGD, Dummy, Random Forest, SVC.

### Bloco 5 — Métricas por classe (4.4)

```python
precisao, recall, f1, suporte = precision_recall_fscore_support(y_train, y_pred, zero_division=0)
```

Sem `average`, a função devolve **um valor por classe**. Os resultados viram uma tabela (modelo, classe).

### Bloco 6 — Pares de confusão

Percorre as previsões dos modelos reais e conta cada par (classe real → classe prevista) em que houve erro.

### Bloco 7 — Observações (Markdown)

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cross_val_predict(modelo, X_train, y_train, cv=cv)` | Uma previsão por amostra, feita por um modelo que não a viu |
| `ConfusionMatrixDisplay.from_predictions(y_real, y_pred, ...)` | Calcula e desenha a matriz a partir das previsões |
| `normalize='true'` | Divide cada linha pelo total da classe real |
| `sample_weight=erros` | Peso 1 para erros e 0 para acertos: a diagonal zera e só os erros contam |
| `values_format='.0%'` | Mostra os valores em porcentagem inteira |
| `display_labels=rotulos` | Nomes das cultivares nos eixos |
| `plot_matrizes_confusao` | Função do notebook: 3 versões por modelo (usada no treino) |
| `previsoes_treino` | Dicionário modelo → previsões fora da amostra |
| `precision_recall_fscore_support` | Métricas e suporte **por classe** |
| `metricas_por_classe` | Tabela modelo × classe |

---

## 4. Conceitos e fórmulas

**Matriz 3 × 3:** o elemento $C_{ij}$ é o número de amostras da classe real $i$ previstas como $j$.

- Diagonal: acertos. Fora da diagonal: confusões.
- Soma da linha $i$: número de amostras da classe $i$. Soma da coluna $j$: número de previsões da classe $j$.

**Relativa (por linha):**

$$R_{ij} = \frac{C_{ij}}{\sum_j C_{ij}} \qquad R_{ii} = Recall_i$$

**Somente erros:** com `sample_weight` = 1 nos erros e 0 nos acertos, a diagonal vira zero e

$$E_{ij} = \frac{C_{ij}}{\sum_{j \neq i} C_{ij}} \quad (i \neq j)$$

é a fração **dos erros da classe $i$** que foram para a classe $j$. Se a classe $i$ não tem erros, a linha toda fica em 0%.

**Por classe, a partir da matriz:**

$$Precision_j = \frac{C_{jj}}{\sum_i C_{ij}} \;(\text{coluna}) \qquad Recall_i = \frac{C_{ii}}{\sum_j C_{ij}} \;(\text{linha})$$

**Arredondamento:** com `values_format='.0%'`, os percentuais de uma linha podem somar 99% ou 101% (por exemplo, 1/57 ≈ 1,75% → "2%", 54/57 ≈ 94,74% → "95%", 2/57 ≈ 3,51% → "4%").

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| `cross_val_predict` no lugar de `predict` no treino | Matrizes fora da amostra, coerentes com a validação cruzada; a referência gerava matrizes otimistas (exigência da especificação) |
| As 3 versões da referência | Decisão do usuário |
| Uma figura por modelo com as 3 versões | Matrizes separadas por modelo e comparação imediata entre as versões; menos células repetidas |
| Função `plot_matrizes_confusao` | 4 modelos × 3 versões = 12 matrizes: repetição relevante |
| Tabela por classe e pares de confusão | A "análise por classe" do plano, no lugar da etapa de limiar dos exercícios anteriores |
| Ordem SGD, Dummy, RF, SVC | Mesma ordem da referência |

---

## 6. Interpretação dos resultados obtidos

**Erros fora da amostra (de 142):**

| Modelo | Erros |
|---|---|
| Dummy | 85 |
| SGD | **1** |
| SVC | 2 |
| Random Forest | 3 |

Esses números são coerentes com as acurácias da Etapa 5 (99,29%, 98,62% e 97,91% como médias por dobra).

**Matrizes (absolutas):**
- **SGD:** diagonal 47 / 56 / 38; único erro: 1 amostra de `class_1` prevista como `class_2`.
- **Dummy:** toda a coluna `class_1` preenchida (47 / 57 / 38). A matriz de erros mostra 100% dos erros de `class_0` e de `class_2` indo para `class_1`, e a linha de `class_1` em 0%, porque nela ele nunca erra.
- **Random Forest:** diagonal 47 / 54 / 38; os 3 erros são da `class_1` (2 → `class_2`, 1 → `class_0`), por isso a matriz de erros mostra 67% e 33% nessa linha.
- **SVC:** 1 amostra de `class_0` → `class_1` e 1 de `class_1` → `class_2`.

**Métricas por classe (fora da amostra):**

| Modelo | Classe | Precisão | Recall | F1 |
|---|---|---|---|---|
| SGD | class_0 | 100,00% | 100,00% | 100,00% |
| SGD | class_1 | 100,00% | 98,25% | 99,12% |
| SGD | class_2 | 97,44% | 100,00% | 98,70% |
| Random Forest | class_0 | 97,92% | 100,00% | 98,95% |
| Random Forest | class_1 | 100,00% | **94,74%** | 97,30% |
| Random Forest | class_2 | 95,00% | 100,00% | 97,44% |
| SVC | class_0 | 100,00% | 97,87% | 98,92% |
| SVC | class_1 | 98,25% | 98,25% | 98,25% |
| SVC | class_2 | 97,44% | 100,00% | 98,70% |
| Dummy | class_1 | 40,14% | 100,00% | 57,29% |
| Dummy | class_0 / class_2 | 0,00% | 0,00% | 0,00% |

**Pares de confusão (modelos reais):**

| Modelo | Real → previsto | Erros |
|---|---|---|
| Random Forest | class_1 → class_2 | 2 |
| SGD | class_1 → class_2 | 1 |
| Random Forest | class_1 → class_0 | 1 |
| SVC | class_0 → class_1 | 1 |
| SVC | class_1 → class_2 | 1 |

**Leitura:**

1. **`class_1` → `class_2` é a confusão mais comum:** são 4 dos 6 erros somados, e aparece nos três modelos.
2. **`class_1` é a cultivar mais difícil:** concentra 5 dos 6 erros. Na análise exploratória, ela se sobrepõe à `class_2` em `proline` e é a mais espalhada em `flavanoids`.
3. **`class_2` nunca foi errada** pelos modelos reais (recall de 100%). A precisão dela é menor que 100% porque recebe amostras de `class_1`.
4. **Dummy:** precisão de `class_1` = 40,14% (a proporção dela no treino) e recall de 100%; nas outras classes, tudo zero.
5. **Comparação:** as diferenças entre os modelos (1, 2 e 3 erros) continuam pequenas. Isso confirma, amostra por amostra, a conclusão da Etapa 5: não há superioridade clara.

---

## 7. Erros comuns e limitações

- **Gerar a matriz do treino com `predict` no próprio treino** (como a referência): a matriz fica otimista (a RF tende a acertar tudo que viu). Corrigido com `cross_val_predict`.
- **Ler a matriz de erros como taxa de erro:** ela mostra a **distribuição** dos erros de cada classe, e não quantos erros houve. Uma linha com 100% pode significar 1 único erro (caso do SGD).
- **Estranhar linhas que somam 101%:** é arredondamento.
- **Inverter eixos:** linhas = real, colunas = prevista (rótulos explícitos nos eixos).
- **Limitação:** com 6 erros no total, a análise de "quais pares se confundem" é baseada em pouquíssimos casos.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que não usar `predict` no treino, como a referência?** Porque o modelo já viu aquelas amostras; a matriz seria otimista. Com `cross_val_predict`, cada amostra é prevista por um modelo que não a viu.
2. **O que mostra a matriz "somente erros"?** Para onde vão os erros de cada classe; a diagonal é zerada.
3. **Qual cultivar é mais difícil?** `class_1`: 5 dos 6 erros dos modelos reais, principalmente confundida com `class_2`.
4. **Por que a precisão da `class_2` não é 100% se o recall é?** Porque ela recebe amostras de `class_1` previstas erradamente como `class_2`.
5. **Quantos erros cada modelo cometeu?** SGD 1, SVC 2 e Random Forest 3, em 142 amostras, fora da amostra.
6. **A diagonal da matriz relativa é o quê?** O recall de cada cultivar.

---

## 9. Como apresentar oralmente

> "Para ver quais cultivares se confundem, gerei as matrizes de confusão do treino com previsões fora da amostra, ao contrário da referência, que usava o próprio treino e por isso ficava otimista. Para cada modelo mostro três versões: a absoluta, a relativa, cuja diagonal é o recall de cada cultivar, e a matriz só de erros, que mostra para onde vão os erros. O SGD errou 1 de 142 amostras, o SVC 2 e a Random Forest 3. O padrão é claro: a `class_1` é a mais difícil e concentra 5 dos 6 erros, quase sempre confundida com a `class_2`. Isso combina com a análise exploratória, em que a `class_1` se sobrepõe à `class_2` em alguns atributos. Já a `class_2` nunca foi errada. O Dummy, como esperado, joga tudo na `class_1`."

Mostrar na tela: a figura do SGD (3 versões), a figura da Random Forest e a tabela de pares de confusão.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Matriz de confusão**

| Item | Para conferir | Link |
|---|---|---|
| `ConfusionMatrixDisplay` | `from_predictions`, `normalize`, `sample_weight`, `values_format`, `colorbar`, `ax` | [scikit-learn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) |
| `confusion_matrix` | Convenção linhas/colunas e normalização | [scikit-learn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| Guia do usuário | Leitura da matriz | [scikit-learn — Confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) |
| Exemplo oficial | Matrizes normalizadas | [scikit-learn — Confusion matrix example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) |
| `cross_val_predict` | Previsões fora da amostra | [scikit-learn — cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |
| `precision_recall_fscore_support` | Métricas por classe (sem `average`) | [scikit-learn — precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `DataFrame.set_index` | Índice (modelo, classe) | [pandas — DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) |
| `DataFrame.value_counts` | Contagem dos pares de confusão | [pandas — DataFrame.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html) |
| `plt.subplots` | Três painéis por figura | [Matplotlib — pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) |
| `fig.suptitle` | Título da figura | [Matplotlib — Figure.suptitle](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html) |
