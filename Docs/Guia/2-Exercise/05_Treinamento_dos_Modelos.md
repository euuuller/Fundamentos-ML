# Etapa 5 — Preparação e treinamento dos três modelos

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **3. Treinamento dos Modelos de Classificação Binária** (3.1 SGD, 3.2 Dummy, 3.3 Random Forest).
Correspondência na referência: seção 3 de `Docs/reference/1-Arquivo.ipynb`, com os mesmos modelos e parâmetros. Adaptação: o `preprocessamento` da Etapa 4 entra no Pipeline do SGD e da RF.

---

## 1. Objetivo e relação com o exercício

Construir e treinar, **só com o treino**, os três modelos definidos para o exercício:

| Modelo | Papel | Entrada |
|---|---|---|
| **SGDClassifier** | Modelo linear | Pipeline: pré-processamento → SGD |
| **DummyClassifier** | Linha de base (não aprende com os atributos) | `X_train` direto |
| **RandomForestClassifier** | Modelo não linear (várias árvores) | Pipeline: pré-processamento → RF |

Nenhuma métrica é calculada aqui. O que se verifica é que os modelos foram montados corretamente e conseguem prever a partir dos **dados brutos** (com texto e ausentes), porque o Pipeline cuida do pré-processamento.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação e pesos balanceados

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

pesos = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
```

Mesma célula do Exercício 01: torna visíveis os pesos que `class_weight='balanced'` aplica dentro do SGD e da RF.

### Bloco 2 — Passageiro de exemplo

```python
indice = np.where(y_train.to_numpy() == 1)[0][0]
passageiro = X_train.iloc[[indice]]
```

Escolhe o **primeiro sobrevivente do treino**, como no Exercício 01, onde o exemplo era o primeiro maligno. `to_numpy()` converte a `Series` em array, para usar posições e não rótulos de índice. `iloc[[indice]]` (colchetes duplos) mantém o formato de tabela (1 × 7) que o `predict` exige.

### Bloco 3 — SGD em Pipeline

```python
sgd_clf = make_pipeline(
    clone(preprocessamento),
    SGDClassifier(class_weight='balanced', random_state=42)
)
sgd_clf.fit(X_train, y_train)
```

No `fit`, o Pipeline:
1. ajusta o pré-processamento em `X_train` (medianas, moda, médias, desvios, categorias);
2. transforma `X_train` na matriz 712 × 10;
3. treina o SGD nessa matriz.

No `predict`, aplica a mesma transformação, com os valores do treino, e depois classifica.

### Bloco 4 — Dummy

```python
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train, y_train)
```

Sem pré-processamento: o Dummy só conta qual classe é a mais frequente em `y_train`.

### Bloco 5 — Random Forest em Pipeline

```python
forest_clf = make_pipeline(
    clone(preprocessamento),
    RandomForestClassifier(class_weight='balanced', random_state=42)
)
forest_clf.fit(X_train, y_train)
```

A RF não precisa de padronização, mas precisa de números sem ausentes, então também recebe o pré-processamento.

### Blocos 6 a 8 — Previsões do exemplo e observações

Cada modelo prevê a classe do mesmo passageiro, e uma célula Markdown resume os resultados.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `compute_class_weight('balanced', ...)` | Pesos por classe usados por `class_weight='balanced'` |
| `make_pipeline(clone(preprocessamento), modelo)` | Pipeline de 2 etapas: `'columntransformer'` e o modelo |
| `clone(preprocessamento)` | Cópia **não ajustada** do pré-processador; cada Pipeline tem a sua |
| `SGDClassifier(class_weight='balanced', random_state=42)` | Parâmetros da referência; demais no padrão (`loss='hinge'`, `penalty='l2'`, `alpha=0.0001`, `max_iter=1000`) |
| `DummyClassifier(strategy="most_frequent", random_state=42)` | Sempre a classe majoritária (o `random_state` não tem efeito nessa estratégia) |
| `RandomForestClassifier(class_weight='balanced', random_state=42)` | Parâmetros da referência; padrão `n_estimators=100`, `max_features='sqrt'`, `bootstrap=True` |
| `.fit(X_train, y_train)` | Ajusta pré-processamento + modelo |
| `.predict(passageiro)` | Pré-processa e classifica |
| `sgd_clf`, `dummy_clf`, `forest_clf` | Modelos treinados, usados nas próximas etapas |
| `indice`, `passageiro` | Exemplo de verificação |

---

## 4. Conceitos e fórmulas

**Pesos balanceados** (treino: 712 passageiros, 439 não sobreviventes, 273 sobreviventes):

$$peso_{0} = \frac{712}{2 \times 439} \approx 0{,}811 \qquad peso_{1} = \frac{712}{2 \times 273} \approx 1{,}304$$

**Classificador linear (SGD):** sobre as 10 colunas transformadas $\mathbf{x}'$,

$$s(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x}' + b \qquad \hat{y} = 1 \text{ se } s(\mathbf{x}) > 0$$

Cada coluna recebe um peso. Por exemplo, `Sex_female` e `Sex_male` têm pesos próprios, o que só é possível porque o *one-hot* as separou.

**Pipeline como um único estimador:** `fit` encadeia `fit_transform` das transformações e `fit` do modelo; `predict` encadeia `transform` e `predict`. Na validação cruzada (Etapa 6), o Pipeline inteiro é **clonado e reajustado em cada dobra**: o pré-processamento também aprende só com as dobras de treino.

**Random Forest:** 100 árvores, cada uma treinada com uma amostra *bootstrap* do treino e, em cada divisão, um subconjunto aleatório das 10 colunas ($\sqrt{10} \approx 3$). A previsão combina as árvores.

**Memorização:** árvores de decisão crescem até separar quase todos os exemplos de treino (`max_depth=None`). Por isso a RF tende a acertar casos **que já viu**, mesmo atípicos, sem que isso indique boa generalização.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Mesmos modelos e parâmetros da referência | Decisão do usuário e consistência com o Exercício 01 |
| Pré-processamento **dentro** do Pipeline | Exigência da especificação: transformações ajustadas só no treino de cada divisão |
| `clone(preprocessamento)` **nos dois** Pipelines | O plano previa o original no SGD e um clone na RF; usar `clone` nos dois deixa o `preprocessamento` como modelo-base não ajustado e evita que os Pipelines compartilhem um mesmo objeto |
| Mesmo pré-processador (com padronização) para a RF | A padronização não altera as divisões das árvores, e um único pré-processador é mais simples de explicar |
| Dummy sem pré-processamento | Ele ignora os atributos |
| `class_weight='balanced'` | Mantido da referência; compensa os 38% de sobreviventes |
| Exemplo = primeiro sobrevivente do treino | Mesma lógica do Exercício 01 (exemplo da classe positiva, onde o Dummy necessariamente erra) |

---

## 6. Interpretação dos resultados obtidos

**Pesos das classes:**

```
Não Sobreviveu (0): peso 0.811
Sobreviveu (1): peso 1.304
```

Um erro em um sobrevivente pesa $1{,}304/0{,}811 \approx 1{,}6$ vez mais, a mesma razão de 439/273.

**Modelos:** as representações mostram `Pipeline(columntransformer → sgdclassifier)`, `DummyClassifier(strategy='most_frequent')` e `Pipeline(columntransformer → randomforestclassifier)`, com os parâmetros esperados.

**Passageiro de exemplo** (posição 0 do treino; índice original 692):

| Pclass | Sex | Age | SibSp | Parch | Fare | Embarked | Classe real |
|---|---|---|---|---|---|---|---|
| 3 | male | NaN | 0 | 0 | 56,4958 | S | **Sobreviveu (1)** |

| Modelo | Classe prevista |
|---|---|
| SGD | Não Sobreviveu (0) |
| Dummy | Não Sobreviveu (0) |
| Random Forest | **Sobreviveu (1)** |

**Leitura:**
- O exemplo é um caso **atípico**: homem da 3ª classe, um perfil que, nos dados da Etapa 2, está associado a baixa sobrevivência (18,89% entre homens; 24,24% na 3ª classe).
- O **SGD** errou: como modelo linear, segue o padrão geral (masculino e 3ª classe empurram a pontuação para "não sobreviveu").
- O **Dummy** errou, como sempre erra os sobreviventes.
- A **Random Forest** acertou, mas este passageiro **estava no treino**: o acerto pode ser apenas memorização. Não indica que ela generaliza melhor.
- Conclusão: é **um único exemplo já visto**. Serve para confirmar que os Pipelines funcionam com dados brutos, incluindo `Age` ausente e texto, mas **não mede desempenho**. A medição honesta começa na Etapa 6.

---

## 7. Erros comuns e limitações

- **Pré-processar fora do Pipeline e depois fazer validação cruzada:** o pré-processamento aprenderia com as dobras de validação, o que é vazamento.
- **Passar ao modelo a matriz já transformada e, depois, `X_test` bruto:** o modelo receberia formatos diferentes. Com o Pipeline, os dois passam pela mesma transformação.
- **Reutilizar o mesmo objeto de pré-processamento em dois Pipelines** sem `clone`: o segundo `fit` sobrescreveria o primeiro. O `clone` evita isso.
- **Concluir algo do exemplo:** um acerto em dado de treino, sobretudo da RF, não indica generalização.
- **Limitação:** hiperparâmetros padrão, sem ajuste, como no Exercício 01.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que o pré-processamento fica dentro do Pipeline?** Para ser ajustado só com os dados de treino de cada divisão (na validação cruzada e no treino final), sem vazamento.
2. **Por que usar `clone`?** Cada Pipeline precisa da sua própria cópia; o original continua sem ajuste.
3. **A RF precisa do pré-processamento?** Precisa da imputação e do *one-hot* (números sem ausentes); a padronização é desnecessária, mas inofensiva.
4. **Por que o Dummy não tem Pipeline?** Ele ignora os atributos; só conta a classe mais frequente.
5. **O que significam os pesos 0,811 e 1,304?** O erro em um sobrevivente pesa cerca de 1,6 vez mais no treino, compensando haver menos sobreviventes.
6. **A RF acertou o exemplo e o SGD errou. A RF é melhor?** Não se pode concluir. O passageiro estava no treino e a RF tende a memorizar; o SGD seguiu o padrão geral para homens da 3ª classe.

---

## 9. Como apresentar oralmente

> "Treinei os três modelos da referência só com o treino. A diferença em relação ao primeiro exercício é que o SGD e a Random Forest recebem o pré-processamento dentro do pipeline: no treino, o pipeline aprende as medianas, a moda e a padronização e depois treina o modelo; na previsão, aplica a mesma transformação. Assim, posso passar os dados brutos, com texto e idades faltando. O Dummy continua sendo a linha de base e prevê sempre 'não sobreviveu'. Os pesos balanceados deram 1,3 para sobreviventes e 0,8 para não sobreviventes. Testei um passageiro do treino que sobreviveu, mas é homem da terceira classe, um caso atípico: o SGD e o Dummy erraram, a Random Forest acertou. Como ele estava no treino, isso pode ser memorização; o desempenho real vem na validação cruzada."

Mostrar na tela: a célula do Pipeline do SGD, os pesos e as três previsões do exemplo.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e pandas 3.0.6. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Modelos**

| Item | Para conferir | Link |
|---|---|---|
| `SGDClassifier` | Parâmetros e `decision_function` | [scikit-learn — SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) |
| Gradiente Descendente Estocástico | Teoria e necessidade de padronizar | [scikit-learn — Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html) |
| `DummyClassifier` | Estratégias de linha de base | [scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) |
| `RandomForestClassifier` | Parâmetros padrão | [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| Florestas aleatórias | Como as árvores são combinadas | [scikit-learn — Random forests](https://scikit-learn.org/stable/modules/ensemble.html#forest) |

**Pipeline e pesos**

| Item | Para conferir | Link |
|---|---|---|
| `Pipeline` | Comportamento de `fit` e `predict` em cadeia | [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| Guia de Pipelines | Encadear pré-processamento e modelo | [scikit-learn — Pipeline: chaining estimators](https://scikit-learn.org/stable/modules/compose.html#pipeline) |
| `clone` | Cópia não ajustada | [scikit-learn — clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) |
| `compute_class_weight` | Fórmula de `'balanced'` | [scikit-learn — compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) |
| `class_weight` | Definição no glossário | [scikit-learn — Glossary: class_weight](https://scikit-learn.org/stable/glossary.html#term-class_weight) |

**NumPy e pandas**

| Item | Para conferir | Link |
|---|---|---|
| `np.where` | Posições dos sobreviventes | [NumPy — numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html) |
| `Series.to_numpy` | Converter para array (posições) | [pandas — Series.to_numpy](https://pandas.pydata.org/docs/reference/api/pandas.Series.to_numpy.html) |
