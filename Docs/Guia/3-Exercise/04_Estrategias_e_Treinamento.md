# Etapa 4 — Estratégias multiclasse e treinamento dos quatro modelos

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seção **3. Classificação Multiclasse** (3.1 SGDClassifier, 3.2 DummyClassifier, 3.3 RandomForest, 3.4 Support Vector Classifier).
Correspondência na referência: seção 3 de `Docs/reference/2-Arquivo.ipynb` (conceitos, os 4 modelos e as pontuações de uma amostra). Adaptações: padronização **dentro de Pipeline** (A1), classe por `classes_[argmax]` (A4), número de classificadores obtido do modelo ou da fórmula, sem a linha artificial `... or 45` (A11), e Dummy `uniform` não incluído (A10).

---

## 1. Objetivo e relação com o exercício

O Exercício 3 tem **3 classes**. Esta etapa:

1. explica as duas famílias de modelos (**nativos** × **estritamente binários**) e as estratégias que permitem usar modelos binários em multiclasse (**OvR** e **OvO**);
2. treina os quatro modelos da referência **só com o treino**;
3. mostra, para **uma mesma amostra**, como cada modelo chega à classe: pontuações, probabilidades ou votos;
4. mostra quantos classificadores binários cada estratégia treina e por que, com 3 classes, os números coincidem.

Nenhuma métrica é calculada aqui; isso começa na Etapa 5.

---

## 2. Código implementado, por blocos

### Bloco 1 — Conceitos (Markdown)

Tabela OvR × OvO (como funciona, nº de classificadores, modelo que usa), fórmulas para N = 3 e a regra de decisão (**maior pontuação**, com `classes_[argmax]`).

### Bloco 2 — Amostra de exemplo

```python
indice_amostra = 0
amostra = X_train.iloc[[indice_amostra]]
```

A **mesma** amostra para os quatro modelos, o que permite comparar como cada um decide. A referência usava amostras diferentes (índices 8 e 1). `iloc[[0]]` mantém o formato de tabela (1 × 13) com os nomes dos atributos.

### Bloco 3 — SGD (OvR) em Pipeline

```python
sgd_clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(class_weight='balanced', max_iter=1000, tol=1e-3, random_state=42)
)
sgd_clf.fit(X_train, y_train)

scores_sgd = sgd_clf.decision_function(amostra)[0]
classe_sgd = sgd_clf.classes_[np.argmax(scores_sgd)]
print(f"Classificadores binários internos (OvR): {sgd_clf[-1].coef_.shape[0]}")
```

- Parâmetros da referência. Na representação impressa, `max_iter=1000` e `tol=1e-3` não aparecem porque **são os valores padrão**.
- `decision_function` devolve **3 pontuações**, uma por classificador binário.
- `sgd_clf[-1]` acessa o último passo do Pipeline (o `SGDClassifier`). `coef_` tem uma linha de pesos por classificador binário, então `coef_.shape[0]` é o número de classificadores, **obtido do próprio modelo**.

### Bloco 4 — Dummy

```python
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.predict_proba(amostra)
```

O `predict_proba` mostra probabilidade 1 para a classe mais frequente e 0 para as outras.

### Bloco 5 — Random Forest (nativo)

```python
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
forest_clf.fit(X_train, y_train)
probas_forest = forest_clf.predict_proba(amostra)[0]
classe_forest = forest_clf.classes_[np.argmax(probas_forest)]
```

Parâmetros da referência (sem `class_weight`, como lá). `n_jobs=-1` só acelera o treino, e o resultado continua reproduzível com `random_state=42`.

### Bloco 6 — SVC (OvO) em Pipeline

```python
svc_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', random_state=42))
svc_clf.fit(X_train, y_train)
scores_svc = svc_clf.decision_function(amostra)[0]
n_classes = len(svc_clf.classes_)
print(f"Classificadores binários internos (OvO): {n_classes * (n_classes - 1) // 2}")
```

- `kernel='rbf'` e `gamma='scale'` são os padrões do SVC, por isso não aparecem na representação impressa.
- O número de classificadores vem da fórmula N(N−1)/2 aplicada ao **número real de classes do modelo** (`len(classes_)`), e não de um valor imposto como na referência.

### Bloco 7 — Observações (Markdown)

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `make_pipeline(StandardScaler(), modelo)` | Padronização aprendida só no treino, aplicada antes do modelo |
| `sgd_clf[-1]` | Último passo do Pipeline (o classificador) |
| `coef_` (SGD) | Matriz 3 × 13: uma linha de pesos por classificador OvR |
| `decision_function(X)` | Pontuações por classe (SGD: OvR; SVC: duelos OvO agregados) |
| `predict_proba(X)` | Probabilidades por classe (RF: proporção de votos; Dummy: 1 para a classe mais frequente) |
| `classes_` | Ordem das classes nas colunas das pontuações: `[0, 1, 2]` |
| `np.argmax(...)` | **Posição** da maior pontuação |
| `classes_[np.argmax(...)]` | **Rótulo** da classe escolhida |
| `class_weight='balanced'` (SGD) | Pesos por classe inversamente proporcionais à frequência |
| `n_estimators=100`, `n_jobs=-1` (RF) | 100 árvores; todos os núcleos |
| `kernel='rbf'`, `gamma='scale'` (SVC) | Fronteiras não lineares; alcance ajustado aos dados |
| `sgd_clf`, `dummy_clf`, `forest_clf`, `svc_clf` | Modelos treinados, usados nas próximas etapas |

---

## 4. Conceitos e fórmulas

**OvR (Um-contra-o-Resto):** para cada classe $k$, um classificador binário "$k$ × resto" gera uma pontuação $s_k(\mathbf{x})$. A decisão é

$$\hat{y} = \arg\max_{k} \; s_k(\mathbf{x})$$

Número de classificadores: $N$.

**OvO (Um-contra-Um):** um classificador para cada par $(a, b)$; cada duelo dá um voto ao vencedor. A decisão é a classe com mais votos. Número de classificadores:

$$\frac{N(N-1)}{2} \quad \Rightarrow \quad N=3: \; 3 \qquad N=10\ \text{(MNIST)}: \; 45$$

Com 3 classes, OvR e OvO têm 3 classificadores cada. A diferença está no que cada um aprende: no OvR, cada classificador usa **todas** as amostras (uma classe contra as outras duas); no OvO, cada um usa **só as amostras de duas classes**.

**Pontuação agregada do SVC:** com `decision_function_shape='ovr'` (padrão), o scikit-learn converte os 3 duelos em uma pontuação por classe:

$$\text{pontuação}_k = \text{votos}_k + \text{termo de confiança (menor que } \tfrac{1}{3} \text{ em módulo)}$$

Por isso, arredondar a pontuação dá o número de duelos vencidos.

**Pesos balanceados do SGD** (cálculo a partir das contagens do treino: 47/57/38 em 142 amostras):

$$peso_c = \frac{142}{3 \times n_c}: \quad class\_0 \approx 1{,}007 \quad class\_1 \approx 0{,}830 \quad class\_2 \approx 1{,}246$$

**Random Forest:** cada árvore vota em uma classe; `predict_proba` é a proporção média de votos, e as 3 probabilidades somam 1.

**Por que `classes_[argmax]`:** `argmax` devolve a posição (0, 1 ou 2) da maior pontuação. Aqui, por coincidência, as classes também são 0, 1 e 2. Em geral, porém, os rótulos poderiam ser textos ou números fora de ordem, e só `classes_` faz a tradução correta. A referência usava `argmax` direto como rótulo (problema "i").

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Parâmetros dos modelos iguais aos da referência | Fidelidade à atividade (decisão do usuário: os 4 modelos) |
| Padronização dentro de Pipeline (SGD e SVC) | Evita vazamento na validação cruzada (Etapa 5); na referência, o *scaler* era ajustado antes |
| Mesma amostra para os 4 modelos | Comparação direta de como cada um decide |
| `classes_[argmax]` | Rótulo correto, independentemente da codificação |
| Nº de classificadores do modelo (`coef_`) ou da fórmula com `len(classes_)` | Sem número imposto (problema "f" da referência) |
| Sem Dummy `uniform` | Decisão do usuário; a referência o treinava sem avaliar |
| RF sem padronização | Divisões por limiar em um atributo não dependem da escala |

---

## 6. Interpretação dos resultados obtidos

**Amostra de exemplo:** classe real **`class_0`**.

| Modelo | Estratégia | Classificadores binários | Pontuações / probabilidades | Classe escolhida |
|---|---|---|---|---|
| SGD | OvR | **3** (`coef_.shape[0]`) | class_0 **35,55**; class_1 −57,17; class_2 −97,19 | **class_0** |
| Dummy | — | — | class_0 0,0; class_1 **1,0**; class_2 0,0 | **class_1** |
| Random Forest | Nativo | — | class_0 **0,98**; class_1 0,01; class_2 0,01 | **class_0** |
| SVC | OvO | **3** (3 × 2 / 2) | class_0 **2,23**; class_1 0,91; class_2 −0,22 | **class_0** |

**Leitura:**

1. **SGD:** só o classificador "`class_0` × resto" deu pontuação positiva (35,55). Os outros dois "disseram não", com pontuações bem negativas. A decisão é clara.
2. **Dummy:** escolhe `class_1` para qualquer amostra, a classe mais frequente do treino (57 de 142), e erra este exemplo.
3. **Random Forest:** 98 das 100 árvores votaram em `class_0`.
4. **SVC:** as pontuações 2,23, 0,91 e −0,22 correspondem a **2, 1 e 0 duelos vencidos**. Uma verificação auxiliar fora do notebook, com `decision_function_shape='ovo'`, mostrou os 3 duelos:
   - `class_0` × `class_1`: valor 1,20, vence a `class_0`;
   - `class_0` × `class_2`: valor 1,02, vence a `class_0`;
   - `class_1` × `class_2`: valor 0,81, vence a `class_1`;

   ou seja, votos [2, 1, 0], e as duas formas dão as mesmas previsões no treino.
5. **Representações impressas:** `max_iter`, `tol`, `kernel` e `gamma` não aparecem porque são os valores padrão. Os modelos foram configurados exatamente como na referência.
6. **Cautela:** é **um exemplo do treino**, já visto pelos modelos. Mostra **como** eles decidem, não **quão bem**.

---

## 7. Erros comuns e limitações

- **Usar `np.argmax` como rótulo:** funciona por acaso quando as classes são 0..N−1 em ordem. O correto é `classes_[argmax]`.
- **Achar que OvR e OvO são iguais porque, com 3 classes, ambos têm 3 classificadores:** os dados usados por cada classificador são diferentes.
- **Padronizar fora do Pipeline:** gera vazamento na validação cruzada.
- **Comparar pontuações entre modelos:** a pontuação do SGD (dezenas), a do SVC (votos + confiança) e as probabilidades da RF estão em escalas diferentes. Só a **ordem dentro de cada modelo** decide a classe.
- **Concluir desempenho a partir de um exemplo:** um acerto em dado de treino não mede generalização.
- **Limitação:** hiperparâmetros padrão da referência, sem ajuste.

---

## 8. Perguntas que o professor poderia fazer

1. **Qual a diferença entre OvR e OvO?** OvR treina um classificador por classe ("classe × resto"); OvO treina um por par de classes e decide por votação.
2. **Quantos classificadores cada estratégia treina aqui?** 3 nos dois casos: N = 3 e N(N−1)/2 = 3. No MNIST seriam 10 e 45.
3. **Então tanto faz OvR ou OvO?** Não: no OvO cada classificador usa só as amostras de duas classes; no OvR, todas.
4. **Como o modelo escolhe a classe?** Pela maior pontuação ou probabilidade; o rótulo vem de `classes_[argmax]`.
5. **Por que o Dummy errou a amostra?** Ele sempre prevê `class_1`, a mais frequente do treino.
6. **O que significam as pontuações do SVC?** Votos dos duelos mais uma pequena confiança: 2,23 indica 2 duelos vencidos pela `class_0`.
7. **Por que a RF não está num Pipeline com padronização?** Não precisa: árvores não dependem da escala.

---

## 9. Como apresentar oralmente

> "Com três cultivares, precisamos de classificação multiclasse. A Random Forest já lida com várias classes. O SGD e o SVC só sabem decidir entre duas opções, então combinam classificadores binários: o SGD usa Um-contra-o-Resto, com um classificador por cultivar; o SVC usa Um-contra-Um, com um classificador por par. Com três classes, os dois treinam três classificadores, mas cada um aprende coisas diferentes. Coloquei a padronização dentro de um pipeline para o SGD e o SVC. Para ver como cada modelo decide, usei a mesma amostra, que é da `class_0`. No SGD, só o classificador da `class_0` deu pontuação positiva; na Random Forest, 98% das árvores votaram nela; no SVC, a `class_0` venceu os dois duelos de que participou. O Dummy chutou `class_1`, a mais frequente. Em todos os casos, a classe escolhida é a de maior pontuação."

Mostrar na tela: a tabela OvR × OvO, as pontuações do SGD e do SVC e as probabilidades da RF.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1 e NumPy 2.5.4. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Estratégias multiclasse**

| Item | Para conferir | Link |
|---|---|---|
| Guia de multiclasse | Visão geral; quais modelos são nativos | [scikit-learn — Multiclass and multioutput algorithms](https://scikit-learn.org/stable/modules/multiclass.html) |
| OvR | Estratégia Um-contra-o-Resto | [scikit-learn — OneVsRest classification](https://scikit-learn.org/stable/modules/multiclass.html#ovr-classification) |
| OvO | Estratégia Um-contra-Um | [scikit-learn — OneVsOne classification](https://scikit-learn.org/stable/modules/multiclass.html#ovo-classification) |
| SVM multiclasse | Como o SVC usa OvO e o `decision_function_shape` | [scikit-learn — SVM: Multi-class classification](https://scikit-learn.org/stable/modules/svm.html#svm-multi-class) |

**Modelos**

| Item | Para conferir | Link |
|---|---|---|
| `SGDClassifier` | Parâmetros, `coef_`, `decision_function` | [scikit-learn — SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) |
| Guia do SGD | OvR no SGD e padronização | [scikit-learn — Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html) |
| `SVC` | `kernel`, `gamma`, `decision_function_shape` | [scikit-learn — SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| `RandomForestClassifier` | `n_estimators`, `n_jobs`, `predict_proba` | [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| `DummyClassifier` | `most_frequent` | [scikit-learn — DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) |

**Pipeline, atributos e NumPy**

| Item | Para conferir | Link |
|---|---|---|
| `make_pipeline` | Encadear padronização e modelo | [scikit-learn — make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) |
| `StandardScaler` | Padronização | [scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) |
| `classes_` | Ordem das classes nas saídas | [scikit-learn — Glossary: classes_](https://scikit-learn.org/stable/glossary.html#term-classes_) |
| `decision_function` | Definição no glossário | [scikit-learn — Glossary: decision_function](https://scikit-learn.org/stable/glossary.html#term-decision_function) |
| `np.argmax` | Posição do maior valor | [NumPy — numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) |
