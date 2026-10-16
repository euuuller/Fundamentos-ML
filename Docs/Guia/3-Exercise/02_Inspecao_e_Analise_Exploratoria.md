# Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seções **2.1. Definição do Alvo** e **2.2. Descrição dos Dados**.
Correspondência na referência: seções 2.1 e 2.2 de `Docs/reference/2-Arquivo.ipynb` (nulos, distribuição das classes em tabela e barras %). Também calcula de verdade a distribuição e as escalas que a célula do Exercício 3 da referência mostrava **sem o código correspondente** (problema "h" do plano).

---

## 1. Objetivo e relação com o exercício

1. **Conferir a codificação do alvo:** quais cultivares correspondem aos códigos 0, 1 e 2.
2. **Medir o balanceamento** entre as 3 classes.
3. **Mostrar que as amostras estão ordenadas por classe**, o que obriga a embaralhar em toda divisão.
4. **Descrever as escalas** dos 13 atributos, o que justifica a padronização para o SGD e o SVC.
5. **Ver quais atributos diferenciam as cultivares**, com médias por classe e boxplots.

---

## 2. Código implementado, por blocos

### Bloco 1 — Conferência da codificação

```python
print(wine.target_names)
for codigo, quantidade in enumerate(np.bincount(y)):
    print(f"Código {codigo} ({wine.target_names[codigo]}): {quantidade} amostras")
```

`target_names[i]` é o nome da classe de código `i`, e `np.bincount` conta cada código.

### Bloco 2 — Nomes legíveis e distribuição

```python
nomes_classes = {codigo: nome for codigo, nome in enumerate(wine.target_names)}
serie_y = pd.Series(y).map(nomes_classes)
distribuicao = pd.DataFrame({'Quantidade': ..., 'Porcentagem': ...})
```

`nomes_classes` (`{0: 'class_0', 1: 'class_1', 2: 'class_2'}`) é construído a partir do próprio dataset, sem digitar os nomes à mão, e será usado em relatórios e matrizes. Segue a tabela de quantidade e porcentagem, no padrão da referência.

### Bloco 3 — Gráfico de barras em %

Mesmo estilo da seção 2.2 da referência (`PercentFormatter`, valor sobre cada barra), com uma cor por cultivar.

### Bloco 4 — Ordem das amostras

```python
print(f"Primeiros 10 rótulos: {y[:10]}")
print(f"Últimos 10 rótulos:   {y[-10:]}")
print(f"Posições em que a classe muda: {np.where(np.diff(y) != 0)[0] + 1}")
plt.scatter(range(len(y)), y, ...)
```

`np.diff(y)` vale zero quando dois rótulos seguidos são iguais; onde é diferente de zero, a classe muda. O gráfico de dispersão (posição × classe) mostra os blocos.

### Bloco 5 — Nulos e escalas

```python
X.isnull().sum().sum()
X.describe().T[['min', 'mean', 'std', 'max']].round(3)
```

### Bloco 6 — Médias por classe e diferença relativa

```python
medias_por_classe = X.groupby(serie_y).mean().T.round(3)
medias_por_classe['Diferença entre classes (em desvios)'] = (
    (medias_por_classe.max(axis=1) - medias_por_classe.min(axis=1)) / X.std()
).round(2)
```

Para cada atributo: a média em cada cultivar e uma medida simples de **quanto as médias se afastam entre as classes**, em unidades de desvio padrão do atributo. Dividir pelo desvio torna atributos de escalas diferentes comparáveis.

### Bloco 7 — Boxplots de três atributos

```python
atributos_destaque = ['flavanoids', 'proline', 'color_intensity']
ax.boxplot(valores_por_classe, tick_labels=list(nomes_classes.values()))
```

Os três atributos foram escolhidos **a partir da tabela do Bloco 6**, como previa o plano: cada um destaca uma cultivar diferente.

### Bloco 8 — Observações (Markdown)

Resume os resultados.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `wine.target_names` | `['class_0' 'class_1' 'class_2']` |
| `np.bincount(y)` | Contagem de cada código |
| `nomes_classes` | **Rótulos legíveis**, usados em todas as próximas etapas |
| `serie_y` | Alvo com nomes, para tabelas e agrupamentos |
| `np.diff`, `np.where` | Detectar as posições em que a classe muda |
| `X.describe()` | Estatísticas de cada atributo |
| `X.groupby(serie_y).mean()` | Média de cada atributo por classe |
| `X.std()` | Desvio padrão de cada atributo (pandas: amostral, `ddof=1`) |
| `ax.boxplot(..., tick_labels=...)` | Boxplot de cada classe |
| `atributos_destaque` | Os três atributos mostrados nos boxplots |

---

## 4. Conceitos

- **Multiclasse e codificação:** os códigos 0/1/2 são apenas rótulos, sem ordem entre cultivares. O que importa é a correspondência com `target_names`.
- **Balanceamento:** proporções desiguais (aqui de 27% a 40%) fazem a classe maior pesar mais na acurácia. As médias **macro** (Etapa 5) dão o mesmo peso a cada classe.
- **Dados ordenados e embaralhamento:** se as amostras estão agrupadas por classe e a divisão não embaralha, uma parte (por exemplo, o teste) pode receber só uma cultivar. O `train_test_split` embaralha por padrão; o `StratifiedKFold` precisa de `shuffle=True`.
- **Padronização:** $z = (x - \mu)/\sigma$. Com `proline` na casa das centenas e `nonflavanoid_phenols` abaixo de 1, modelos baseados em distância ou gradiente (SVC, SGD) seriam dominados pelos atributos de maior escala sem a padronização.
- **Diferença entre classes em desvios:**

  $$d_j = \frac{\max_c \bar{x}_{j,c} - \min_c \bar{x}_{j,c}}{s_j}$$

  em que $\bar{x}_{j,c}$ é a média do atributo $j$ na classe $c$ e $s_j$ o seu desvio padrão. É uma medida **descritiva e simples** de quanto um atributo varia entre as cultivares. Não é um teste estatístico e não considera a sobreposição dentro das classes.
- **Boxplot:** a caixa vai do 1º ao 3º quartil, a linha central é a mediana, as hastes mostram o alcance típico e os círculos são valores atípicos.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Conferir `target_names` antes de qualquer uso | Exigência da especificação; no Exercício 01, a codificação estava invertida |
| Não recodificar o alvo | Os códigos já correspondem aos nomes, na ordem de `target_names` |
| Nomes construídos do próprio dataset | Evita erro de digitação e mantém a correspondência exata |
| Mostrar a ordem das amostras | Torna visível o motivo de embaralhar; a referência não tratava disso |
| `describe` com todos os atributos | Reproduz de forma verificável o que a referência mostrava sem código, e com todos os 13 atributos, não só 4 |
| Médias por classe + diferença em desvios | Escolha dos atributos dos boxplots baseada nos dados, e não arbitrária |
| `flavanoids`, `proline`, `color_intensity` | Têm diferenças grandes entre classes (2,20; 1,89; 1,86 desvios) e cada um destaca uma cultivar diferente; `proline` também ilustra a maior escala |
| EDA no conjunto completo | É descritiva; nada é aprendido para os modelos. A padronização será ajustada só no treino, dentro do Pipeline |

---

## 6. Interpretação dos resultados obtidos

**Codificação e distribuição:**

```
['class_0' 'class_1' 'class_2']
Código 0 (class_0): 59 amostras
Código 1 (class_1): 71 amostras
Código 2 (class_2): 48 amostras
```

| Classe | Quantidade | Porcentagem |
|---|---|---|
| class_0 | 59 | 33,15% |
| class_1 | 71 | 39,89% |
| class_2 | 48 | 26,97% |

- A codificação está correta e não precisa ser alterada.
- As classes são **levemente desbalanceadas**: a maior (`class_1`) tem cerca de 1,5 vez a menor (`class_2`). Um modelo que sempre prevê `class_1` acertaria ≈40%. É isso que o Dummy vai mostrar.

**Ordem das amostras:**

```
Primeiros 10 rótulos: [0 0 0 0 0 0 0 0 0 0]
Últimos 10 rótulos:   [2 2 2 2 2 2 2 2 2 2]
Posições em que a classe muda: [ 59 130]
```

As posições 0–58 são `class_0`, 59–129 são `class_1` e 130–177 são `class_2`, em **três blocos contínuos**, como mostra o gráfico. Sem embaralhar, por exemplo, os últimos 20% (o teste) seriam quase todos `class_2`.

**Nulos:** 0 em `X` e em `y`.

**Escalas (seleção):**

| Atributo | mín. | média | desvio | máx. |
|---|---|---|---|---|
| proline | 278,00 | 746,893 | 314,907 | 1680,00 |
| magnesium | 70,00 | 99,742 | 14,282 | 162,00 |
| alcalinity_of_ash | 10,60 | 19,495 | 3,340 | 30,00 |
| alcohol | 11,03 | 13,001 | 0,812 | 14,83 |
| hue | 0,48 | 0,957 | 0,229 | 1,71 |
| nonflavanoid_phenols | 0,13 | 0,362 | 0,124 | 0,66 |

O desvio de `proline` é cerca de 2500 vezes o de `nonflavanoid_phenols`. A padronização é necessária para o SGD e o SVC.

**Médias por classe (ordenadas pela diferença entre classes):**

| Atributo | class_0 | class_1 | class_2 | Diferença (desvios) |
|---|---|---|---|---|
| flavanoids | 2,982 | 2,081 | 0,781 | **2,20** |
| od280/od315_of_diluted_wines | 3,158 | 2,785 | 1,684 | 2,08 |
| proline | 1115,712 | 519,507 | 629,896 | 1,89 |
| total_phenols | 2,840 | 2,259 | 1,679 | 1,86 |
| color_intensity | 5,528 | 3,087 | 7,396 | 1,86 |
| alcohol | 13,745 | 12,279 | 13,154 | 1,81 |
| hue | 1,062 | 1,056 | 0,683 | 1,66 |
| … | | | | |
| magnesium | 106,339 | 94,549 | 99,312 | 0,83 |
| ash | 2,456 | 2,245 | 2,437 | 0,77 |

**Boxplots:**
- **`flavanoids`:** decresce de `class_0` (mediana ≈2,98) para `class_1` (≈2,03) e `class_2` (≈0,68). A `class_2` fica bem abaixo das outras, e a `class_1` é a mais espalhada (de 0,57 a 5,08).
- **`proline`:** a `class_0` se destaca com valores altos (mediana ≈1095), enquanto `class_1` (≈495) e `class_2` (≈628) se sobrepõem bastante.
- **`color_intensity`:** a `class_1` tem valores baixos (mediana ≈2,9), a `class_2` altos (≈7,55) e a `class_0` intermediários (≈5,4).
- **Leitura conjunta:** cada atributo isolado separa bem **uma** cultivar das outras, mas há sobreposição. Combinando-os, como os modelos farão com os 13 atributos, as três cultivares tendem a ficar bem separadas. A confirmação vem com os modelos, nas próximas etapas.

---

## 7. Erros comuns e limitações

- **Dividir sem embaralhar:** com os dados em blocos, o teste ficaria concentrado em uma cultivar.
- **Ler os códigos 0/1/2 como ordem:** são categorias, sem ordem.
- **Escolher atributos "no olho":** aqui a escolha veio de uma medida calculada.
- **Tratar a diferença em desvios como teste estatístico:** é uma medida descritiva, que olha só as médias.
- **Padronizar o conjunto inteiro agora:** seria vazamento; a padronização ficará no Pipeline (Etapa 4).
- **Limitação:** a EDA olha um atributo de cada vez; correlações entre atributos (por exemplo, `flavanoids` × `total_phenols`) não foram analisadas, para manter a etapa simples.

---

## 8. Perguntas que o professor poderia fazer

1. **A codificação do alvo precisou ser alterada?** Não; 0, 1 e 2 já correspondem a `class_0`, `class_1` e `class_2`.
2. **As classes são balanceadas?** Levemente desbalanceadas: 39,89%, 33,15% e 26,97%.
3. **Por que é importante embaralhar?** As amostras vêm em blocos por classe (a classe muda nas posições 59 e 130); sem embaralhar, treino e teste teriam proporções muito diferentes de cada cultivar.
4. **Por que padronizar?** As escalas variam muito (prolina até 1680; fenóis não flavonoides até 0,66), e o SGD e o SVC são sensíveis à escala.
5. **Como você escolheu os três atributos dos boxplots?** Pela diferença entre as médias das classes, em desvios padrão: os escolhidos estão entre os maiores, e cada um destaca uma cultivar diferente.
6. **Algum atributo sozinho separa as três cultivares?** Não totalmente; cada um separa bem uma delas, mas há sobreposição entre as outras.

---

## 9. Como apresentar oralmente

> "Primeiro conferi o alvo: os códigos 0, 1 e 2 correspondem às cultivares `class_0`, `class_1` e `class_2`, então não precisei recodificar. São 59, 71 e 48 vinhos, uma distribuição levemente desbalanceada. Um detalhe importante: o dataset vem ordenado por classe, em três blocos, então toda divisão precisa embaralhar os dados. Não há valores ausentes, mas as escalas são muito diferentes: a prolina chega a 1680 e os fenóis não flavonoides não passam de 0,66, por isso vou padronizar para o SGD e o SVC. Por fim, calculei quanto a média de cada atributo varia entre as cultivares: os flavonoides separam bem a `class_2`, a prolina destaca a `class_0` e a intensidade de cor distingue a `class_1`. Cada atributo sozinho separa uma cultivar; combinados, devem separar as três."

Mostrar na tela: a conferência do alvo, o gráfico da ordem das amostras, a tabela de médias por classe e os três boxplots.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Dataset e conceitos**

| Item | Para conferir | Link |
|---|---|---|
| `load_wine` | `target_names` e conteúdo | [scikit-learn — load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| Padronização | Por que modelos sensíveis à escala precisam dela | [scikit-learn — Standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) |

**NumPy**

| Item | Para conferir | Link |
|---|---|---|
| `np.bincount` | Contagem por código | [NumPy — numpy.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) |
| `np.diff` | Diferença entre elementos consecutivos | [NumPy — numpy.diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) |
| `np.where` | Posições em que a classe muda | [NumPy — numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html) |

**pandas**

| Item | Para conferir | Link |
|---|---|---|
| `Series.map` | Códigos → nomes | [pandas — Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) |
| `Series.value_counts` | Contagens e proporções | [pandas — Series.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) |
| `isnull` | Checagem de nulos | [pandas — DataFrame.isnull](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) |
| `describe` | Escalas dos atributos | [pandas — DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) |
| `groupby` | Médias por classe | [pandas — DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) |
| `std` | Desvio padrão (`ddof=1`) | [pandas — DataFrame.std](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.std.html) |

**Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `PercentFormatter` | Eixo em porcentagem | [Matplotlib — ticker.PercentFormatter](https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.PercentFormatter) |
| `plt.scatter` | Gráfico da ordem das amostras | [Matplotlib — pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) |
| `ax.boxplot` | Boxplots por classe (`tick_labels`) | [Matplotlib — Axes.boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html) |
