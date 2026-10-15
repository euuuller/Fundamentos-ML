# Etapa 4 — Preparação e treinamento dos três modelos

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **3. Treinamento dos Modelos de Classificação Binária** (3.1 SGD, 3.2 Dummy, 3.3 Random Forest).
Correspondência na referência: seção 3 de `Docs/reference/1-Arquivo.ipynb`, com os mesmos modelos e parâmetros. A única adaptação é o `Pipeline` no SGD.

---

## 1. Objetivo e relação com o exercício

O enunciado pede para treinar **os mesmos 3 modelos** da referência. Esta etapa os constrói e os treina **só com o conjunto de treino**:

| Modelo | Papel no exercício |
|---|---|
| **SGDClassifier** (com padronização) | Modelo linear, rápido e simples |
| **DummyClassifier** | Linha de base: mostra o que se obtém **sem aprender nada** |
| **RandomForestClassifier** | Modelo não linear baseado em várias árvores |

Ainda **não há avaliação de desempenho**; isso começa na Etapa 5. O que se verifica aqui é que os modelos foram montados corretamente e conseguem prever.

---

## 2. Código implementado, por blocos

### Bloco 1 — Pesos de `class_weight='balanced'`

```python
from sklearn.utils.class_weight import compute_class_weight

pesos = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
for classe, peso in zip([0, 1], pesos):
    print(f"{nomes_classes[classe]}: peso {peso:.3f}")
```

Calcula explicitamente os pesos que o SGD e a Random Forest usarão internamente, para que o efeito de `class_weight='balanced'` fique visível e possa ser explicado.

### Bloco 2 — SGD dentro de um `Pipeline`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

sgd_clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(class_weight='balanced', random_state=42)
)
sgd_clf.fit(X_train, y_train)
```

`make_pipeline` encadeia duas etapas. No `fit`, o `StandardScaler` aprende média e desvio no treino e transforma os dados, e depois o `SGDClassifier` é treinado nos dados transformados. No `predict`, os dados novos passam pelo **mesmo** `transform` (com a média e o desvio do treino) antes de chegar ao classificador.

Diferença em relação à referência: lá, `scaler.fit_transform(X_train)` era feito antes, e o `sgd_clf` recebia `X_train_scaled`. Isso funciona para um único treino, mas na validação cruzada a padronização teria sido calculada com **todas** as dobras, inclusive a de validação, o que é vazamento. Com o `Pipeline`, a padronização é refeita dentro de cada dobra automaticamente.

### Bloco 3 — Efeito da padronização (gráfico)

```python
scaler = sgd_clf.named_steps['standardscaler']
X_train_scaled = scaler.transform(X_train)
atributo_idx = list(cancer.feature_names).index('mean area')
# histogramas: original × padronizado
print(scaler.mean_[atributo_idx], scaler.scale_[atributo_idx])
```

Mesmo gráfico da referência (antes × depois da padronização), aplicado ao atributo `mean area`. O `scaler` é **o que está dentro do Pipeline**, já ajustado no treino. `X_train_scaled` serve só para visualização; os modelos não o usam.

### Bloco 4 — Previsão de exemplo (SGD)

```python
indice = np.where(y_train == 1)[0][0]
print(f"Classe real:  {nomes_classes[y_train[indice]]}")
print(f"Classe prevista (SGD): {nomes_classes[sgd_clf.predict(X_train[[indice]])[0]]}")
```

A referência usava o índice fixo 16. Aqui foi escolhido o **primeiro tumor maligno do treino**, porque com um exemplo benigno os três modelos acertariam e a comparação não mostraria nada. `X_train[[indice]]` (colchetes duplos) mantém o formato 2D (1 × 30) que o `predict` exige.

### Bloco 5 — DummyClassifier

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train, y_train)
```

O "treino" só conta qual é a classe mais frequente em `y_train`. Não precisa de padronização, porque ignora os atributos.

### Bloco 6 — RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
forest_clf.fit(X_train, y_train)
```

Recebe `X_train` sem padronização. Usa os demais parâmetros com o valor padrão, como na referência.

### Blocos 7 e 8 — Previsões de exemplo (Dummy e RF) e observações

Repetem a previsão do mesmo tumor para cada modelo. Uma célula Markdown final resume os resultados.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `compute_class_weight('balanced', classes, y)` | Calcula os pesos por classe que `class_weight='balanced'` aplica |
| `make_pipeline(*etapas)` | Cria um `Pipeline` e dá nomes automáticos às etapas (`'standardscaler'`, `'sgdclassifier'`) |
| `sgd_clf.named_steps['standardscaler']` | Acessa o scaler já ajustado dentro do Pipeline |
| `scaler.mean_`, `scaler.scale_` | Média e desvio padrão aprendidos **no treino**, um por atributo |
| `SGDClassifier(class_weight='balanced', random_state=42)` | Parâmetros da referência. Os demais ficam no padrão da versão instalada: `loss='hinge'` (SVM linear), `penalty='l2'`, `alpha=0.0001`, `max_iter=1000`, `tol=0.001`, `learning_rate='optimal'` |
| `DummyClassifier(strategy="most_frequent")` | Sempre prevê a classe majoritária do treino. O `random_state` não tem efeito nessa estratégia; foi mantido por fidelidade à referência |
| `RandomForestClassifier(class_weight='balanced', random_state=42)` | Padrões da versão instalada: `n_estimators=100` árvores, `criterion='gini'`, `max_depth=None`, `max_features='sqrt'`, `bootstrap=True` |
| `.fit(X, y)` | Treina o modelo e retorna o próprio objeto (por isso o notebook exibe a representação do modelo) |
| `.predict(X)` | Retorna a classe prevista (0 ou 1) para cada linha de `X` |
| `sgd_clf`, `dummy_clf`, `forest_clf` | Os três modelos treinados, usados nas próximas etapas |
| `indice` | Posição do tumor de exemplo em `X_train` |

---

## 4. Conceitos e fórmulas

**Pesos balanceados:**

$$peso_{classe} = \frac{n_{amostras}}{n_{classes} \times n_{amostras\ da\ classe}}$$

Com o treino da Etapa 3 (455 amostras, 285 benignas e 170 malignas):

$$peso_{0} = \frac{455}{2 \times 285} \approx 0{,}798 \qquad peso_{1} = \frac{455}{2 \times 170} \approx 1{,}338$$

Durante o treino, o erro em cada amostra é multiplicado pelo peso da sua classe. Assim, as duas classes contribuem igualmente para o erro total, mesmo com quantidades diferentes de amostras.

**Padronização (*z-score*):**

$$z = \frac{x - \mu_{treino}}{\sigma_{treino}}$$

$\mu$ e $\sigma$ são aprendidos no treino e reutilizados em qualquer dado novo.

**Classificador linear (SGD):**

$$s(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = \sum_{j=1}^{30} w_j x_j + b \qquad \hat{y} = \begin{cases} 1 & \text{se } s(\mathbf{x}) > 0 \\ 0 & \text{caso contrário} \end{cases}$$

O SGD ajusta os pesos $\mathbf{w}$ e o viés $b$ aos poucos, uma amostra por vez, na direção que reduz a perda (*hinge*, com regularização L2). A pontuação $s(\mathbf{x})$ é a `decision_function`, e o limiar 0 será estudado na Etapa 6.

**Random Forest:** cada árvore é treinada com uma amostra *bootstrap* (sorteio com reposição) do treino e, em cada divisão, considera só um subconjunto aleatório dos atributos ($\sqrt{30} \approx 5$). A previsão final combina as árvores: a classe com maior probabilidade média. A diversidade entre as árvores reduz a variância (*overfitting*) de uma árvore isolada.

**Linha de base (*baseline*):** um ponto de comparação trivial. Se um modelo real não supera o Dummy, ele não aprendeu nada útil.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Mesmos modelos e parâmetros da referência | Exigência do enunciado ("os mesmos 3 modelos") |
| `Pipeline` no SGD | Exigência da especificação: ajustar transformações apenas no treino de cada divisão e evitar vazamento na validação cruzada |
| Sem padronização no Dummy e na RF | O Dummy ignora os atributos; a RF não depende da escala. Padronizar seria inútil e acrescentaria complexidade |
| `class_weight='balanced'` | Mantido da referência. Faz o treino dar mais peso aos malignos (minoritários), o que favorece o recall, objetivo do exercício |
| Hiperparâmetros padrão (sem ajuste) | A referência não faz ajuste de hiperparâmetros, e o exercício pede os mesmos modelos. Ajustar exigiria uma busca com validação cruzada que não faz parte do escopo |
| Exemplo maligno na previsão | Mostra de forma didática a diferença entre um modelo que aprende e a *baseline* |
| `compute_class_weight` explícito | Torna visível um parâmetro que, de outra forma, ficaria "escondido" dentro dos modelos |

---

## 6. Interpretação dos resultados obtidos

**Pesos das classes:**

```
Benigno (0): peso 0.798
Maligno (1): peso 1.338
```

Um erro em um tumor maligno pesa $1{,}338 / 0{,}798 \approx 1{,}7$ vez mais que um erro em um benigno, a mesma razão entre o número de benignos e de malignos no treino (285/170).

**Modelos ajustados** (representações exibidas pelo notebook):

```
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('sgdclassifier', SGDClassifier(class_weight='balanced', random_state=42))])
DummyClassifier(random_state=42, strategy='most_frequent')
RandomForestClassifier(class_weight='balanced', random_state=42)
```

**Padronização de `mean area`** (aprendida no treino):

- média = **659,578**; desvio padrão = **360,022**;
- no gráfico, a forma da distribuição é **idêntica** antes e depois, com o mesmo pico e a mesma cauda à direita. Só o eixo muda: de cerca de 143–2500 para cerca de −1,4 a 5,1 desvios padrão.

A padronização **não muda a informação**, só a escala. Por isso não "melhora os dados"; ela torna o treino do SGD equilibrado entre os atributos.

**Previsão do exemplo** (índice 0 do treino, um tumor **maligno**):

| Modelo | Classe prevista |
|---|---|
| SGD | Maligno (1) |
| Dummy | Benigno (0) |
| Random Forest | Maligno (1) |

O Dummy errou, como esperado: ele responde "Benigno" para **qualquer** tumor. Atenção: é **um único exemplo**, que os modelos **já viram no treino**. Isso confirma que os modelos funcionam, mas **não mede desempenho**. Acertar dados de treino é fácil, sobretudo para a Random Forest, que tende a memorizá-los. A medição honesta começa na Etapa 5, com validação cruzada.

---

## 7. Erros comuns e limitações

- **Padronizar fora do Pipeline e depois fazer validação cruzada:** a padronização usaria informação da dobra de validação, e as métricas sairiam levemente otimistas.
- **Chamar `fit_transform` no teste:** o teste deve passar só por `transform` com os parâmetros do treino. Com o Pipeline, basta `sgd_clf.predict(X_test)`, e isso é feito automaticamente.
- **Passar `X_train[indice]` (1D) ao `predict`:** gera erro de formato. É preciso `X_train[[indice]]` ou `reshape(1, -1)`.
- **Concluir algo a partir da previsão de exemplo:** um acerto em dado de treino não indica generalização.
- **Achar que `class_weight` "resolve" o desbalanceamento:** ele só muda a importância dos erros no treino. O equilíbrio final entre recall e precisão ainda depende do **limiar** (Etapa 6).
- **Limitação:** os hiperparâmetros não foram ajustados; os resultados refletem as configurações padrão da referência.
- **Reprodutibilidade:** `random_state=42` fixa a aleatoriedade do SGD (ordem das amostras) e da RF (*bootstrap* e sorteio de atributos).

---

## 8. Perguntas que o professor poderia fazer

1. **Por que usar `Pipeline`, se a referência não usou?** Para que a padronização seja aprendida só nos dados de treino de cada dobra da validação cruzada, sem vazamento. A especificação exige isso.
2. **Por que só o SGD é padronizado?** Ele é um modelo linear treinado por gradiente e é sensível à escala. A RF compara um atributo por vez com um limiar, e o Dummy ignora os atributos.
3. **O que `class_weight='balanced'` faz aqui?** Multiplica o erro dos malignos por 1,338 e o dos benignos por 0,798, compensando o fato de haver menos malignos.
4. **Por que o Dummy previu "Benigno" para um tumor maligno?** Porque ele sempre prevê a classe mais frequente do treino (Benigno), sem olhar os atributos.
5. **O acerto do SGD e da RF no exemplo prova que são bons?** Não. É um único exemplo já visto no treino. O desempenho será medido com validação cruzada.
6. **Por que o `random_state` no Dummy?** Para manter a configuração da referência; na estratégia `most_frequent` ele não tem efeito.
7. **Qual a função de perda do SGD?** `hinge` (padrão), equivalente a uma SVM linear, com regularização L2 (`alpha=0.0001`).

---

## 9. Como apresentar oralmente

> "Treinei os três modelos da referência só com o conjunto de treino. O SGD é um classificador linear sensível à escala, então coloquei o `StandardScaler` junto com ele em um `Pipeline`. Essa é a minha adaptação em relação à referência: garante que a padronização seja aprendida só com os dados de treino, inclusive dentro de cada dobra da validação cruzada. O gráfico mostra que a padronização muda a escala da área média, mas não a forma da distribuição. O Dummy é a linha de base e sempre prevê a classe mais frequente, que é benigno. A Random Forest combina 100 árvores e não precisa de padronização. O SGD e a Random Forest usam `class_weight='balanced'`, que dá peso 1,34 aos malignos e 0,80 aos benignos. Testei um tumor maligno do treino: o SGD e a Random Forest acertaram e o Dummy errou. Mas isso é só uma verificação de funcionamento; o desempenho real vem na validação cruzada."

Mostrar na tela: a célula do `Pipeline`, o gráfico antes × depois e as três previsões do exemplo.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Modelos**

| Item | Para conferir | Link |
|---|---|---|
| `SGDClassifier` | Parâmetros (`loss`, `penalty`, `alpha`, `class_weight`, `random_state`) e métodos (`decision_function`) | [scikit-learn — SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) |
| Gradiente Descendente Estocástico | Teoria, funções de perda e a recomendação de padronizar os dados | [scikit-learn — Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html) |
| `DummyClassifier` | Estratégias (`most_frequent`, `stratified`, `uniform`...) | [scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) |
| Estimadores *dummy* | Uso como linha de base | [scikit-learn — Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) |
| `RandomForestClassifier` | Parâmetros (`n_estimators`, `max_features`, `bootstrap`, `class_weight`) | [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| Florestas aleatórias | Como as árvores são combinadas e por que reduzem a variância | [scikit-learn — Random forests](https://scikit-learn.org/stable/modules/ensemble.html#forest) |
| Árvores de decisão | Funcionamento das divisões por atributo | [scikit-learn — Decision Trees](https://scikit-learn.org/stable/modules/tree.html) |

**Pré-processamento e Pipeline**

| Item | Para conferir | Link |
|---|---|---|
| `make_pipeline` | Criação do Pipeline com nomes automáticos | [scikit-learn — make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) |
| `Pipeline` | `named_steps`, comportamento de `fit`/`predict` | [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| Guia de Pipelines | Por que encadear transformações e modelo | [scikit-learn — Pipeline: chaining estimators](https://scikit-learn.org/stable/modules/compose.html#pipeline) |
| `StandardScaler` | Atributos `mean_` e `scale_` | [scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) |
| Vazamento de dados | O problema que o Pipeline evita | [scikit-learn — Common pitfalls: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |

**Pesos das classes**

| Item | Para conferir | Link |
|---|---|---|
| `compute_class_weight` | A fórmula usada por `'balanced'` | [scikit-learn — compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) |
| `class_weight` | Definição no glossário | [scikit-learn — Glossary: class_weight](https://scikit-learn.org/stable/glossary.html#term-class_weight) |

**NumPy e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `np.where` | Encontrar os índices dos tumores malignos | [NumPy — numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html) |
| `ax.hist()` | Histogramas antes × depois | [Matplotlib — Axes.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html) |
