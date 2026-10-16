# Etapa 1 — Importações e obtenção dos dados

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seções **1. Observar o Panorama Geral** e **2. Obtenção dos Dados** (início).
Correspondência na referência: seção 1 de `Docs/reference/2-Arquivo.ipynb` (`plt.rc` e panorama) e a célula do **EXERCÍCIO 3** (enunciado e `load_wine()`).

---

## 1. Objetivo e relação com o exercício

Antes de qualquer análise, é preciso **definir o problema** e **carregar os dados**. Esta etapa:

1. enquadra o problema: classificação **multiclasse** com 3 cultivares, 13 entradas, métricas do enunciado;
2. configura a aparência dos gráficos **uma única vez**, para todo o notebook;
3. carrega o *Wine Recognition Dataset* e separa as entradas (`X`) da saída (`y`), do jeito que o enunciado indica;
4. lista os 13 atributos físico-químicos.

---

## 2. Código implementado, por blocos

### Bloco 1 — Configuração dos gráficos

```python
import matplotlib.pyplot as plt
plt.rc('font', size=14)
...
```

Mesmos valores da referência. Na referência, o `plt.rc` é alterado de novo no meio do notebook (`size=9`/`10`, antes das matrizes de confusão), o que muda todos os gráficos seguintes. Aqui ele é configurado **só uma vez** (adaptação A7 do plano).

### Bloco 2 — Panorama Geral (Markdown)

Reproduz o enunciado do Exercício 3 e acrescenta duas explicações:
- em multiclasse, precisão, recall e F1 são **por classe**, resumidos pela **média macro**;
- não há curva ROC nem ajuste de limiar, porque a classe prevista é a de **maior pontuação**. A referência citava "Curva ROC" sem calculá-la, e o enunciado do Exercício 3 não a pede.

### Bloco 3 — Carregamento

```python
from sklearn.datasets import load_wine
wine = load_wine()
wine.keys()
```

`load_wine()` lê o dataset embutido no scikit-learn, sem download e sem pasta `datasets/`, ao contrário do Titanic. O retorno é um `Bunch`.

### Bloco 4 — Entradas e saída

```python
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
```

É a mesma construção da célula do enunciado na referência. `X` vira um `DataFrame` com o nome de cada atributo nas colunas; `y` continua um array NumPy com os códigos das classes.

### Bloco 5 — Nomes dos atributos e observações

Lista numerada dos 13 atributos e uma célula Markdown de observações.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `plt.rc(grupo, **params)` | Parâmetros padrão do matplotlib, definidos uma vez |
| `load_wine()` | Carrega o dataset Wine como `Bunch` |
| `wine` | `Bunch` com `data`, `target`, `frame`, `target_names`, `DESCR`, `feature_names` |
| `wine.data` | Array 178 × 13 com as medições |
| `wine.target` | Array com o código da cultivar (conferido na Etapa 2) |
| `wine.feature_names` | Nomes dos 13 atributos |
| `X` | `DataFrame` (178, 13): entradas |
| `y` | `ndarray` (178,): saída |

---

## 4. Conceitos

- **Classificação multiclasse:** há **três ou mais** classes possíveis, e cada amostra pertence a **exatamente uma**. É diferente da classificação **multirrótulo**, em que uma amostra pode ter vários rótulos ao mesmo tempo.
- **Supervisionado:** a cultivar de cada amostra é conhecida e serve de rótulo para o treino.
- **MISO:** 13 entradas numéricas → 1 saída (a cultivar).
- **Os 13 atributos** (tradução livre dos nomes do dataset):

| Atributo | Significado |
|---|---|
| `alcohol` | Teor alcoólico |
| `malic_acid` | Ácido málico |
| `ash` | Cinzas |
| `alcalinity_of_ash` | Alcalinidade das cinzas |
| `magnesium` | Magnésio |
| `total_phenols` | Fenóis totais |
| `flavanoids` | Flavonoides |
| `nonflavanoid_phenols` | Fenóis não flavonoides |
| `proanthocyanins` | Proantocianidinas |
| `color_intensity` | Intensidade de cor |
| `hue` | Matiz (tonalidade) |
| `od280/od315_of_diluted_wines` | Razão de absorbância OD280/OD315 dos vinhos diluídos |
| `proline` | Prolina (aminoácido) |

- **Decisão multiclasse:** cada modelo gera uma pontuação (ou probabilidade) para cada uma das 3 classes, e a classe prevista é a de maior valor. Como não existe um "limiar" único, como no binário, não há ajuste de limiar neste exercício.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| `plt.rc` uma única vez | Evita o problema da referência (fontes alteradas no meio do notebook) |
| `X` como `DataFrame`, `y` como array | Exatamente como no enunciado; o `DataFrame` facilita a análise por nome de atributo |
| Importar só o que a etapa usa | Estilo da referência e dos exercícios anteriores; evita antecipar etapas |
| Não reproduzir agora as saídas extras da célula do Wine (distribuição e escalas) | Na referência, elas aparecem sem o código que as gera; aqui serão calculadas de fato na Etapa 2 |
| Sem `assert` de versões | Proibido pela especificação |

---

## 6. Interpretação dos resultados obtidos

Saídas do notebook executado:

```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])
X: DataFrame com formato (178, 13)
y: ndarray com formato (178,)
Entrada  0: alcohol
...
Entrada 12: proline
```

- **178 amostras** e **13 atributos**, todos numéricos e contínuos. Diferentemente do Titanic, não há texto para codificar.
- O conjunto é **pequeno**. Na divisão 80/20, o teste terá poucas dezenas de amostras, o que vai exigir cautela na interpretação e torna a validação cruzada estratificada importante.
- `y` ainda está na codificação original. A correspondência entre os códigos e os nomes das cultivares (`target_names`) é conferida na Etapa 2.

---

## 7. Erros comuns e limitações

- **Assumir a ordem ou a codificação das classes sem conferir:** será feito na Etapa 2.
- **Alterar `plt.rc` no meio do notebook:** muda silenciosamente todos os gráficos seguintes.
- **Confundir multiclasse com multirrótulo:** aqui cada vinho tem uma única cultivar.
- **Limitação:** 178 amostras de uma única região. Os resultados valem só para este exercício.

---

## 8. Perguntas que o professor poderia fazer

1. **O que torna este problema multiclasse?** Há 3 classes possíveis, e cada vinho pertence a exatamente uma delas.
2. **Por que MISO?** São 13 atributos de entrada para uma única saída (a cultivar).
3. **Por que não há curva ROC nem ajuste de limiar?** O enunciado não os pede, e em multiclasse a decisão é a classe de maior pontuação, sem um limiar único.
4. **Por que `X` é um `DataFrame`?** Para seguir o enunciado e trabalhar com os atributos pelo nome.
5. **Precisa baixar o dataset?** Não: ele vem embutido no scikit-learn.
6. **O que significa `(178, 13)`?** 178 vinhos (linhas) e 13 medições químicas (colunas).

---

## 9. Como apresentar oralmente

> "O terceiro exercício é uma classificação multiclasse: identificar qual das três cultivares produziu um vinho a partir de 13 medições químicas, como álcool, flavonoides e prolina. Cada vinho pertence a exatamente uma cultivar. As métricas são acurácia, precisão, recall, F1 e matriz de confusão; como há três classes, precisão, recall e F1 são calculados por classe e resumidos pela média macro. Não há curva ROC nem ajuste de limiar, porque o modelo escolhe sempre a classe de maior pontuação. Carreguei o dataset do próprio scikit-learn: são 178 amostras e 13 atributos numéricos, um conjunto pequeno, o que vai pesar nas próximas escolhas."

Mostrar na tela: o Panorama Geral, a saída `(178, 13)` e a lista de atributos.

---

## 10. Documentação oficial

Links para a versão *stable* ou atual de cada documentação (conferidos em 2026-10-12). O ambiente usa Python 3.14, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Dataset**

| Item | Para conferir | Link |
|---|---|---|
| `load_wine()` | Parâmetros e conteúdo do `Bunch` | [scikit-learn — load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| Descrição do dataset | Atributos, estatísticas e distribuição das classes (mesmo texto do `DESCR`) | [scikit-learn — Wine recognition dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset) |
| Fonte original | Repositório UCI | [UCI ML Repository — Wine](https://archive.ics.uci.edu/dataset/109/wine) |
| `Bunch` | Objeto retornado | [scikit-learn — sklearn.utils.Bunch](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html) |

**Conceitos**

| Item | Para conferir | Link |
|---|---|---|
| Multiclasse no scikit-learn | Multiclasse × multirrótulo; estratégias OvR e OvO (base da Etapa 4) | [scikit-learn — Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html) |
| Glossário | Definição de *multiclass* | [scikit-learn — Glossary: multiclass](https://scikit-learn.org/stable/glossary.html#term-multiclass) |

**pandas, NumPy, Matplotlib e Python**

| Item | Para conferir | Link |
|---|---|---|
| `pd.DataFrame` | Criação com `columns=` | [pandas — DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) |
| `ndarray` | Tipo de `y` | [NumPy — numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) |
| `plt.rc` | Configuração dos gráficos | [Matplotlib — pyplot.rc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html) |
| `rcParams` | Parâmetros configuráveis | [Matplotlib — Customizing with rcParams](https://matplotlib.org/stable/users/explain/customizing.html) |
| `enumerate()` | Lista numerada dos atributos | [Python — enumerate](https://docs.python.org/3/library/functions.html#enumerate) |
