# Etapa 1 — Importações e obtenção dos dados

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seções **1. Observar o Panorama Geral** e **2. Obtenção dos Dados** (início).
Correspondência na referência: seção 1 e início da seção 2 de `Docs/reference/1-Arquivo.ipynb`, além da célula do enunciado do Exercício 1.

---

## 1. Objetivo e relação com o exercício

Antes de treinar qualquer modelo, é preciso **definir o problema** e **ter os dados em mãos**. Esta etapa:

1. enquadra o problema (que tipo de aprendizado é, qual a tarefa, como o desempenho será medido);
2. configura a aparência dos gráficos, como na referência;
3. carrega o dataset *Breast Cancer Wisconsin (Diagnostic)* e separa as entradas (`X`) da saída (`y_original`);
4. lista os 30 atributos de entrada (a célula que o enunciado já trazia).

Ela também prepara a discussão central do exercício: **por que o recall é a métrica prioritária**.

---

## 2. Código implementado, por blocos

### Bloco 1 — Configuração dos gráficos

```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

Copiado da referência. Define tamanhos de fonte padrão para todos os gráficos do notebook, para que fiquem legíveis e com aparência uniforme.

### Bloco 2 — Panorama Geral (Markdown)

Descreve o dataset, o objetivo e o enquadramento: **supervisionado**, **classificação binária**, **MISO**, **em lote**, métricas com **recall como prioridade**.

### Bloco 3 — Carregamento do dataset

```python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()
```

`load_breast_cancer()` lê o dataset que vem **dentro** do scikit-learn, sem internet e sem cache, ao contrário do `fetch_openml` usado no MNIST da referência. O retorno é um objeto `Bunch`, que funciona como um dicionário.

### Bloco 4 — Entradas e saída

```python
X, y_original = cancer.data, cancer.target
print(type(X))
print(X.shape)
print(y_original.shape)
```

`X` é a matriz de atributos e `y_original` é o vetor de diagnósticos **na codificação original do dataset**. O nome `y_original`, e não `y`, é proposital: a codificação ainda será conferida e, se necessário, convertida na Etapa 2.

### Bloco 5 — Nomes dos atributos

```python
for i, nome in enumerate(cancer.feature_names):
    print(f"Entrada {i:2d}: {nome}")
```

Mesma célula do enunciado do Exercício 1.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | O que é |
|---|---|
| `plt.rc(grupo, **params)` | Altera os parâmetros padrão do matplotlib (*runtime configuration*) para todo o notebook |
| `load_breast_cancer()` | Carrega o dataset embutido. Sem argumentos, devolve um `Bunch` com arrays NumPy |
| `cancer` | `Bunch` com as chaves obtidas: `data`, `target`, `frame`, `target_names`, `DESCR`, `feature_names`, `filename`, `data_module` |
| `cancer.data` → `X` | `numpy.ndarray` de formato **(569, 30)**: 569 tumores × 30 atributos |
| `cancer.target` → `y_original` | `numpy.ndarray` de formato **(569,)**: um rótulo por tumor |
| `cancer.feature_names` | Nomes das 30 colunas de `X` |
| `cancer.target_names` | Nomes das classes, usado na Etapa 2 para conferir a codificação |
| `cancer.DESCR` | Descrição textual do dataset (origem, atributos, estatísticas) |
| `cancer.frame` | Fica `None` com o carregamento padrão; só é preenchido com `as_frame=True` |

---

## 4. Conceitos

- **Aprendizado supervisionado:** cada amostra de treino vem com a resposta correta (o diagnóstico), e o modelo aprende a mapear atributos → rótulo.
- **Classificação binária:** a saída é uma entre duas classes (Maligno / Benigno).
- **MISO** (*Multiple Input, Single Output*): 30 atributos de entrada geram uma única saída. A referência classificou o MNIST como SISO, mas aqui o caso é claramente MISO (como também está descrito no Exercício 2 da referência).
- **Aprendizado em lote** (*batch*): o modelo é treinado com todo o conjunto de treino de uma vez; não chegam dados novos continuamente.
- **Os 30 atributos:** o `DESCR` informa que são **10 medidas** dos núcleos celulares (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma resumida de **3 formas**:
  - `mean …` (atributos 0–9): média;
  - `… error` (10–19): **erro padrão** da medida;
  - `worst …` (20–29): média dos **três maiores** valores.
  Por exemplo: atributo 0 = *mean radius*, 10 = *radius error*, 20 = *worst radius*.
- **Por que recall:** as fórmulas e a comparação entre métricas são assunto da Etapa 5. Por enquanto basta a ideia: um **falso negativo** (maligno dado como benigno) pode atrasar o tratamento, enquanto um **falso positivo** gera exames adicionais. O primeiro erro é mais grave, então a capacidade de **encontrar os malignos** (recall) é priorizada.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Manter a estrutura e o `plt.rc` da referência | A especificação pede que a ordem e o estilo da referência sejam preservados |
| Sem verificações de versão com `assert` (a referência tem) | Proibido pela especificação; o ambiente foi conferido na Etapa 0 |
| Importar só o que a etapa usa (`matplotlib`, `load_breast_cancer`) | Segue o estilo da referência, que importa cada recurso na célula em que é usado, e evita antecipar etapas. `numpy` e `pandas` entram na Etapa 2. *(Pequeno ajuste em relação ao plano, que listava os dois aqui.)* |
| Arrays NumPy (sem `as_frame=True`) | Igual ao MNIST da referência (`as_frame=False`); o `DataFrame` será montado na Etapa 2, só para a análise exploratória |
| Nome `y_original` | Deixa explícito que o alvo **ainda não** está na codificação exigida (Maligno = 1). A conferência e a conversão são feitas na Etapa 2 |

---

## 6. Interpretação dos resultados obtidos

Saídas do notebook executado:

- `cancer.keys()` → `dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])`
- `type(X)` → `<class 'numpy.ndarray'>`
- `X.shape` → `(569, 30)`: **569 amostras** e **30 atributos**
- `y_original.shape` → `(569,)`: um rótulo por amostra
- Lista com os 30 atributos, de `Entrada 0: mean radius` até `Entrada 29: worst fractal dimension`

Leitura: o dataset é **pequeno** (569 amostras, contra 70 000 do MNIST). Isso pesa em escolhas futuras: validação cruzada com mais dobras (Etapa 5) e cautela ao interpretar diferenças pequenas entre os modelos. Os 30 atributos são numéricos e contínuos, então não é preciso codificar variáveis categóricas.

---

## 7. Erros comuns e limitações

- **Assumir que 1 = maligno.** No `load_breast_cancer`, isso **não** é garantido; a conferência é obrigatória e é feita na Etapa 2. Usar o alvo sem conferir inverteria o significado de recall e precisão.
- **Confundir `… error` com erro do modelo.** Nesses atributos, *error* é o **erro padrão** da medida na imagem, não um erro de previsão.
- **Esperar um `DataFrame` em `cancer.frame`.** Ele fica `None` sem `as_frame=True`.
- **Limitação:** os dados são de um único centro (Wisconsin) e de 1995 (`DESCR`). Os resultados valem só para este exercício acadêmico.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que este problema é supervisionado?** Porque cada tumor tem o diagnóstico conhecido, que serve de rótulo para o treino.
2. **Por que MISO e não SISO?** São 30 atributos de entrada para uma única saída (o diagnóstico).
3. **O que significa `X.shape = (569, 30)`?** 569 tumores (linhas), cada um descrito por 30 atributos (colunas).
4. **Qual a diferença entre `mean radius`, `radius error` e `worst radius`?** Média, erro padrão e média dos três maiores valores do raio dos núcleos na mesma imagem.
5. **Por que você não chamou o alvo de `y`?** Porque ainda não foi conferido se 1 representa maligno; isso é feito na próxima etapa antes de definir `y`.
6. **Por que o recall é a métrica prioritária?** Porque o falso negativo (deixar passar um maligno) é o erro mais caro em triagem.

---

## 9. Como apresentar oralmente

> "Comecei enquadrando o problema: é um aprendizado supervisionado de classificação binária, com 30 entradas e uma saída, treinado em lote. Como o erro mais grave é dizer que um tumor maligno é benigno, o recall é a métrica principal. Depois carreguei o dataset de câncer de mama do próprio scikit-learn: são 569 tumores com 30 atributos numéricos, que são 10 medidas dos núcleos celulares, cada uma em média, erro padrão e pior valor. Guardei o rótulo como `y_original` porque, antes de usá-lo, vou conferir na próxima etapa se a classe 1 realmente representa o tumor maligno."

Mostrar na tela: a célula de Panorama Geral, a saída `(569, 30)` e a lista de atributos.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, matplotlib 3.11.2 e Python 3.14. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Dataset**

| Item | Para conferir | Link |
|---|---|---|
| `load_breast_cancer()` | Parâmetros (`return_X_y`, `as_frame`) e o que o `Bunch` retornado contém | [scikit-learn — load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| Descrição do dataset | Atributos, estatísticas, distribuição das classes (mesmo texto do `DESCR`) | [scikit-learn — Breast cancer Wisconsin (diagnostic) dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) |
| Fonte original | Repositório UCI, de onde o dataset foi copiado | [UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| `Bunch` | O objeto "dicionário com acesso por atributo" (`cancer.data`, `cancer['data']`) | [scikit-learn — sklearn.utils.Bunch](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html) |

**NumPy e Python**

| Item | Para conferir | Link |
|---|---|---|
| `ndarray` | O tipo de `X` e `y_original` | [NumPy — numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) |
| `.shape` | Significado de `(569, 30)` | [NumPy — ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html) |
| `enumerate()` | Laço com índice da lista de atributos | [Python — enumerate](https://docs.python.org/3/library/functions.html#enumerate) |
| *f-strings* | Formatação `f"Entrada {i:2d}: {nome}"` | [Python — f-strings](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) |

**Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `plt.rc()` | Como alterar os parâmetros padrão | [Matplotlib — pyplot.rc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html) |
| `rcParams` | Lista e explicação dos parâmetros configuráveis (`font`, `axes`, `xtick`...) | [Matplotlib — Customizing with style sheets and rcParams](https://matplotlib.org/stable/users/explain/customizing.html) |

**Conceitos**

| Item | Para conferir | Link |
|---|---|---|
| Aprendizado supervisionado | Visão geral dos modelos supervisionados do scikit-learn | [scikit-learn — Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) |
