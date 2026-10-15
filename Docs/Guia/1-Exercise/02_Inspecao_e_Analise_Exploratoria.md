# Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seções **2.1. Definição do Alvo** e **2.2. Descrição dos Dados**.
Correspondência na referência: seções 2.1 (Observação dos Primeiros Dados) e 2.2 (Descrição dos Dados) de `Docs/reference/1-Arquivo.ipynb`.

---

## 1. Objetivo e relação com o exercício

O enunciado define **Maligno = 1** e **Benigno = 0**. Esta etapa:

1. **comprova** como o dataset codifica as classes e converte o alvo para a codificação do exercício;
2. verifica se há dados ausentes;
3. resume os atributos (estatísticas descritivas);
4. mede o **desbalanceamento** entre as classes;
5. mostra que os atributos têm **escalas muito diferentes**.

Sem o passo 1, todas as métricas seguintes ficariam com o significado invertido: o "recall" mediria a detecção de **benignos**.

---

## 2. Código implementado, por blocos

### Bloco 1 — Conferência da codificação original

```python
import numpy as np

print(cancer.target_names)

for codigo, quantidade in enumerate(np.bincount(y_original)):
    print(f"Código {codigo} ({cancer.target_names[codigo]}): {quantidade} amostras")
```

`target_names[i]` é o nome da classe de código `i`, e `np.bincount` conta quantas vezes cada código aparece.

### Bloco 2 — Inversão do alvo e tabela de conferência

```python
import pandas as pd

y = 1 - y_original

conferencia = pd.DataFrame({
    'Classe': cancer.target_names[y_original],
    'Código original': y_original,
    'Código novo (y)': y
})
conferencia.value_counts().to_frame('Quantidade')
```

`1 - y_original` troca 0↔1 numa única operação vetorizada. A tabela agrupa cada combinação (nome, código original, código novo) e conta as amostras, provando que **todo** maligno virou 1 e **todo** benigno virou 0.

### Bloco 3 — Dados nulos

```python
df = pd.DataFrame(X, columns=cancer.feature_names)
print(f"... em X: {df.isnull().sum().sum()}")
print(f"... em y: {pd.Series(y).isnull().sum()}")
```

O `DataFrame` dá nome às colunas e permite usar os métodos do pandas. A soma dupla conta os nulos de todas as colunas.

### Bloco 4 — Estatísticas descritivas

```python
df.iloc[:, :10].describe().T.round(3)
```

Mostra só os 10 atributos de **média** (colunas 0–9), para a tabela ficar legível; os grupos `error` e `worst` são as mesmas medidas resumidas de outra forma. O `.T` transpõe a tabela (um atributo por linha).

### Bloco 5 — Distribuição das classes (tabela)

```python
nomes_classes = {0: 'Benigno (0)', 1: 'Maligno (1)'}
serie_y = pd.Series(y).map(nomes_classes)
distribuicao = pd.DataFrame({
    'Quantidade': serie_y.value_counts(),
    'Porcentagem': serie_y.value_counts(normalize=True).map('{:.2%}'.format)
})
```

Mesmo padrão da referência (quantidade absoluta e relativa lado a lado), agora com rótulos legíveis.

### Bloco 6 — Distribuição das classes (gráfico)

Gráfico de barras em porcentagem, no estilo da seção 2.2 da referência: `PercentFormatter` no eixo Y, valor escrito sobre cada barra e grade horizontal.

### Bloco 7 — Histogramas por classe

```python
atributos = ['mean area', 'mean smoothness']
...
ax.hist(df.loc[y == 0, atributo], ..., label='Benigno (0)')
ax.hist(df.loc[y == 1, atributo], ..., label='Maligno (1)')
```

`df.loc[y == 0, atributo]` filtra as linhas de uma classe (máscara booleana) e seleciona uma coluna. As duas classes ficam sobrepostas e semitransparentes (`alpha=0.6`).

### Bloco 8 — Observações (Markdown)

Resume no notebook o que as saídas mostraram.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `cancer.target_names` | `['malignant' 'benign']`: nome da classe de cada código |
| `np.bincount(v)` | Conta as ocorrências de cada inteiro não negativo em `v` |
| `y = 1 - y_original` | **Alvo oficial do exercício** (1 = maligno). Todas as etapas seguintes usam `y` |
| `df` | `DataFrame` (569 × 30) com os atributos nomeados. Usado só para a exploração; os modelos continuarão recebendo `X` |
| `DataFrame.value_counts()` | Conta combinações únicas de linhas |
| `describe()` | count, mean, std, min, quartis e max de cada coluna |
| `value_counts(normalize=True)` | Frequência relativa (proporção) |
| `nomes_classes`, `serie_y` | Rótulos legíveis das classes, só para tabelas e gráficos |
| `mtick.PercentFormatter(1.0)` | Mostra 0–1 como 0%–100% |

---

## 4. Conceitos

- **Classe positiva:** no scikit-learn, `precision_score`, `recall_score` e `f1_score` usam `pos_label=1` por padrão. O que vale 1 é a classe "positiva", cujos acertos e erros são medidos. Por isso, **Maligno precisa ser 1**.
- **Codificação de rótulos:** é uma convenção arbitrária do dataset. O código numérico não tem significado por si; o que importa é o mapeamento para o nome da classe.
- **Desbalanceamento:** proporção desigual entre classes. Aqui ele é **moderado** (37,26% × 62,74%), muito menor que o do MNIST "é 5" da referência (≈9% × 91%). Mesmo assim, um classificador que sempre responde "benigno" acertaria ≈62,7% das vezes sem detectar nenhum câncer. É isso que o *DummyClassifier* vai mostrar nas próximas etapas.
- **Escala dos atributos:** modelos lineares treinados por gradiente (SGD) são sensíveis à escala. Atributos com valores grandes dominam o cálculo do gradiente e da regularização. A padronização (*z-score*) leva todos à média 0 e desvio 1:

  $$z = \frac{x - \mu}{\sigma}$$

  Árvores de decisão (Random Forest) comparam limiares em um atributo de cada vez, então a escala não altera as divisões.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Conferir `target_names` antes de inverter | A especificação exige confirmar a codificação original; inverter "de memória" seria uma suposição |
| `1 - y_original`, e não um `map`/`where` | Operação mais simples possível para trocar 0↔1, fácil de explicar |
| Tabela de conferência com `value_counts()` | Prova, com saída do notebook, que nenhuma amostra ficou com rótulo errado |
| `describe()` só dos atributos `mean` | Mantém a tabela legível; a mensagem (escalas diferentes) já aparece aí |
| Histogramas de `mean area` e `mean smoothness` | Foram escolhidos por terem escalas extremas (centenas/milhares × centésimos) e mostrarem dois níveis de sobreposição entre as classes |
| EDA "simples" | A especificação pede análise exploratória simples; não há seleção nem engenharia de atributos, porque o exercício usa os 30 atributos nos 3 modelos |
| Exploração no conjunto completo (antes da divisão) | Aqui é só descritiva: nenhum parâmetro é aprendido a partir dela. O `StandardScaler` (que aprende média e desvio) só será ajustado no treino, dentro do `Pipeline` (Etapa 4) |

---

## 6. Interpretação dos resultados obtidos

**Codificação original** (saída do Bloco 1):

```
['malignant' 'benign']
Código 0 (malignant): 212 amostras
Código 1 (benign): 357 amostras
```

Confirma que o dataset usa **0 = maligno**, o inverso do exercício.

**Conferência após a inversão** (Bloco 2):

| Classe | Código original | Código novo (y) | Quantidade |
|---|---|---|---|
| benign | 1 | 0 | 357 |
| malignant | 0 | 1 | 212 |

Só existem essas duas combinações, ou seja, a conversão está correta para todas as 569 amostras.

**Nulos:** 0 em `X` e 0 em `y`. Nenhum tratamento de dados ausentes é necessário (o `DESCR` também informa *Missing Attribute Values: None*).

**Estatísticas descritivas** (atributos de média, valores arredondados):

| Atributo | mínimo | média | máximo |
|---|---|---|---|
| mean area | 143,500 | 654,889 | 2501,000 |
| mean perimeter | 43,790 | 91,969 | 188,500 |
| mean radius | 6,981 | 14,127 | 28,110 |
| mean smoothness | 0,053 | 0,096 | 0,163 |
| mean fractal dimension | 0,050 | 0,063 | 0,097 |

O desvio padrão vai de **351,914** (`mean area`) a **0,007** (`mean fractal dimension`), uma diferença de cerca de 50 000 vezes. Isso reforça a necessidade de padronizar os dados para o SGD.

**Distribuição das classes:**

| Classe | Quantidade | Porcentagem |
|---|---|---|
| Benigno (0) | 357 | 62,74% |
| Maligno (1) | 212 | 37,26% |

O gráfico de barras mostra o mesmo (62,7% × 37,3%). O desbalanceamento moderado justifica: (a) a **estratificação** na divisão (Etapa 3) e na validação cruzada; (b) o `class_weight='balanced'` dos modelos (Etapa 4); (c) não confiar só na acurácia.

**Histogramas:**
- `mean area`: os benignos se concentram em valores mais baixos (em torno de 300–700), e os malignos se espalham por valores maiores (até ≈2500). Há uma faixa de **sobreposição** (aproximadamente entre 400 e 900), na qual o atributo sozinho não decide a classe.
- `mean smoothness`: escala de 0,05 a 0,16; as duas distribuições se **sobrepõem bastante**, com os malignos levemente deslocados para valores maiores.
- Conclusão: nenhum atributo isolado separa perfeitamente as classes. Os modelos precisam combinar vários atributos.

---

## 7. Erros comuns e limitações

- **Não conferir a codificação.** É o erro mais grave possível neste exercício: os números de recall pareceriam ótimos, mas mediriam a detecção de benignos.
- **Inverter duas vezes.** Depois desta etapa, o alvo é **sempre** `y`; `y_original` não deve mais ser usado nos modelos.
- **Achar que 37% "não é desbalanceado".** O desbalanceamento é moderado, mas ainda faz a acurácia de um classificador trivial ficar em ≈63%.
- **Ajustar transformações no dataset inteiro.** Calcular estatísticas para *exibir* é seguro; *ajustar* um `StandardScaler` antes da divisão seria vazamento. Isso será evitado com o `Pipeline`.
- **Limitação:** foram analisados só 2 dos 30 atributos em histograma. É uma visão ilustrativa, não uma análise completa de relevância dos atributos.

---

## 8. Perguntas que o professor poderia fazer

1. **Como você sabe que 0 era maligno?** Pelo `target_names = ['malignant', 'benign']`, em que a posição 0 é maligno, e pelas contagens (212 malignos, 357 benignos), que batem com o `DESCR`.
2. **Por que inverter, em vez de usar `pos_label=0` nas métricas?** Porque o exercício define Maligno = 1, e todas as funções (relatórios, curvas, `class_weight`, matriz de confusão) passam a tratar o maligno como positivo sem configuração extra em cada chamada. Isso reduz a chance de erro.
3. **O dataset é desbalanceado?** Moderadamente: 37,26% malignos. Um modelo que sempre diz "benigno" teria ≈63% de acurácia e recall zero.
4. **Por que olhar a escala dos atributos?** Porque o SGD é sensível à escala. `mean area` vai até 2501, e `mean smoothness` não passa de 0,163. Sem padronização, atributos grandes dominariam o treino.
5. **A Random Forest precisa de padronização?** Não. Cada divisão de uma árvore usa um limiar em um único atributo, e mudar a escala não altera a ordem dos valores.
6. **Fazer a EDA antes da divisão não é vazamento?** Não, porque aqui nada é aprendido e usado pelo modelo; é só descrição. Vazamento seria ajustar transformações ou escolhas do modelo com dados de teste.

---

## 9. Como apresentar oralmente

> "Antes de usar o rótulo, conferi a codificação: o `target_names` mostra que, no dataset original, 0 é maligno e 1 é benigno, o contrário do que o exercício pede. Como as métricas tratam a classe 1 como positiva, inverti o alvo com `1 - y_original` e montei uma tabela de conferência que prova que os 212 malignos viraram 1 e os 357 benignos viraram 0. Não há dados nulos. As classes são moderadamente desbalanceadas, 37% de malignos, então vou estratificar a divisão e não vou confiar só na acurácia. Por fim, os histogramas mostram que os atributos têm escalas muito diferentes, com a área chegando a 2500 e a suavidade abaixo de 0,2. Isso justifica padronizar os dados para o SGD. Também mostram que nenhum atributo sozinho separa as classes."

Mostrar na tela: a saída de `target_names`, a tabela de conferência, o gráfico de proporções e os histogramas.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Alvo e codificação**

| Item | Para conferir | Link |
|---|---|---|
| `target_names` | Campo do `Bunch` com o nome de cada classe | [scikit-learn — load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| `np.bincount()` | Contagem de cada valor inteiro | [NumPy — numpy.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) |
| `1 - y_original` | Operação vetorizada: o escalar é aplicado a cada elemento (*broadcasting*) | [NumPy — Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| `pos_label=1` | Por que a classe positiva das métricas precisa ser o maligno | [scikit-learn — recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) |

**pandas**

| Item | Para conferir | Link |
|---|---|---|
| `pd.DataFrame` | Criação com `columns=` | [pandas — DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) |
| `DataFrame.value_counts()` | Contagem de combinações de linhas (tabela de conferência) | [pandas — DataFrame.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html) |
| `Series.value_counts()` | Contagem por classe e `normalize=True` | [pandas — Series.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) |
| `Series.to_frame()` | Transforma a contagem em tabela com nome de coluna | [pandas — Series.to_frame](https://pandas.pydata.org/docs/reference/api/pandas.Series.to_frame.html) |
| `isnull()` | Detecção de valores ausentes | [pandas — DataFrame.isnull](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) |
| `describe()` | Estatísticas descritivas (count, mean, std, quartis) | [pandas — DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) |
| `iloc` | Seleção por posição (`df.iloc[:, :10]`) | [pandas — DataFrame.iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html) |
| `loc` + máscara booleana | Filtrar uma classe (`df.loc[y == 0, atributo]`) | [pandas — DataFrame.loc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html) · [Boolean indexing](https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing) |
| `Series.map()` | Troca códigos por rótulos legíveis | [pandas — Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) |
| `Series.plot(kind='bar')` | Gráfico de barras direto da série | [pandas — Series.plot](https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html) |

**Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `PercentFormatter` | Eixo Y em porcentagem | [Matplotlib — ticker.PercentFormatter](https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.PercentFormatter) |
| `ax.text()` | Valor escrito acima de cada barra | [Matplotlib — Axes.text](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| `plt.subplots()` | Figura com dois gráficos lado a lado | [Matplotlib — pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) |
| `ax.hist()` | Histograma (`bins`, `alpha`, `label`) | [Matplotlib — Axes.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html) |

**Conceitos citados (usados nas próximas etapas)**

| Item | Para conferir | Link |
|---|---|---|
| Padronização (*z-score*) | Por que modelos lineares precisam de dados padronizados | [scikit-learn — Standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) |
| `StandardScaler` | A classe que fará a padronização dentro do `Pipeline` | [scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) |
