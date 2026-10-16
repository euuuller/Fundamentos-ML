# Etapa 4 — Pré-processamento com `ColumnTransformer`

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **2.4. Pré-processamento com `ColumnTransformer`**.
Correspondência na referência: **não há equivalente**. O MNIST é todo numérico e sem ausentes, e o Exercício 01 também. É a principal novidade do Exercício 02 (adaptação A3 do plano).

---

## 1. Objetivo e relação com o exercício

Os modelos do scikit-learn exigem uma matriz **numérica e sem ausentes**. Os dados do Titanic têm:

- valores **ausentes** em `Age` (137 no treino) e `Embarked` (2 no treino);
- atributos de **texto**: `Sex` (`female`/`male`) e `Embarked` (`C`/`Q`/`S`);
- atributos numéricos em **escalas diferentes** (`Fare` até 512; `Parch` até 6).

Esta etapa monta o `preprocessamento`, que resolve os três problemas e será colocado **dentro do Pipeline** de cada modelo na Etapa 5. Para entender o que ele faz, uma cópia é ajustada **só no treino** e inspecionada.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Tabela com o tratamento de cada grupo de colunas e a justificativa de colocar tudo dentro do Pipeline.

### Bloco 2 — Montagem do `ColumnTransformer`

```python
atributos_numericos = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
atributos_categoricos = ['Sex', 'Embarked']

num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                             OneHotEncoder(handle_unknown='ignore'))

preprocessamento = ColumnTransformer([
    ('num', num_pipeline, atributos_numericos),
    ('cat', cat_pipeline, atributos_categoricos)
])
```

- Cada "mini-pipeline" encadeia duas transformações: **preencher** os ausentes e depois **padronizar** ou **codificar**.
- O `ColumnTransformer` recebe uma lista de trios `(nome, transformador, colunas)`, aplica cada transformador às suas colunas e **concatena** os resultados lado a lado.
- Aqui o objeto ainda **não aprendeu nada**: está só configurado.

### Bloco 3 — Inspeção: ajuste de uma cópia no treino

```python
prep_inspecao = clone(preprocessamento).fit(X_train)
num_ajustado = prep_inspecao.named_transformers_['num']
...
pd.DataFrame({
    'Mediana (imputação)': num_ajustado.named_steps['simpleimputer'].statistics_,
    'Média (padronização)': num_ajustado.named_steps['standardscaler'].mean_,
    'Desvio (padronização)': num_ajustado.named_steps['standardscaler'].scale_
}, index=atributos_numericos)
```

- `clone` cria uma cópia **não ajustada**, para que o `preprocessamento` original continue "limpo" para os Pipelines dos modelos.
- `named_transformers_` (com `_` no fim, porque existe só após o `fit`) dá acesso a cada parte ajustada, e `named_steps` dá acesso a cada etapa de um mini-pipeline.
- `statistics_` contém as medianas ou modas aprendidas; `mean_` e `scale_`, a média e o desvio da padronização.

### Bloco 4 — Valores aprendidos para os categóricos

Imprime a moda usada na imputação e as categorias encontradas pelo `OneHotEncoder` (`categories_`) para `Sex` e `Embarked`.

### Bloco 5 — Resultado da transformação

```python
X_train_prep = prep_inspecao.transform(X_train)
nomes_colunas_prep = prep_inspecao.get_feature_names_out()
```

Mostra o tipo do resultado, o formato antes e depois, a contagem de ausentes após a transformação e os nomes das colunas geradas.

### Bloco 6 — Exemplo antes × depois

```python
posicao = X_train['Age'].isnull().to_numpy().argmax()
print(X_train.iloc[[posicao]].to_string())
print(pd.DataFrame(X_train_prep[[posicao]], columns=nomes_colunas_prep).round(3).to_string())
```

`argmax()` sobre a máscara booleana devolve a **primeira** posição com `True`, ou seja, o primeiro passageiro do treino sem idade. A mesma linha é mostrada antes e depois do pré-processamento.

### Bloco 7 — Observações (Markdown)

Resume os resultados.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `atributos_numericos`, `atributos_categoricos` | Listas de colunas de cada grupo (reutilizadas nas próximas etapas) |
| `SimpleImputer(strategy='median')` | Substitui ausentes pela **mediana** da coluna (aprendida no `fit`) |
| `SimpleImputer(strategy='most_frequent')` | Substitui ausentes pela **moda** (valor mais frequente) |
| `StandardScaler()` | $z = (x - \mu)/\sigma$, com $\mu$ e $\sigma$ do treino |
| `OneHotEncoder(handle_unknown='ignore')` | Uma coluna 0/1 por categoria; categoria nova (não vista no treino) vira tudo 0 |
| `make_pipeline(...)` | Encadeia as transformações de cada grupo |
| `ColumnTransformer([...])` | Aplica cada transformador às suas colunas e junta o resultado |
| `preprocessamento` | **Objeto oficial**, não ajustado, que entra no Pipeline dos modelos (Etapa 5) |
| `clone(...)` | Cópia não ajustada de um estimador |
| `prep_inspecao` | Cópia ajustada no treino, **só para inspeção** |
| `named_transformers_`, `named_steps` | Acesso às partes ajustadas |
| `statistics_`, `mean_`, `scale_`, `categories_` | Valores aprendidos no `fit` |
| `get_feature_names_out()` | Nomes das colunas geradas (`num__Age`, `cat__Sex_male`...) |
| `X_train_prep`, `nomes_colunas_prep` | Treino transformado (712 × 10) e nomes das colunas, só para inspeção |

---

## 4. Conceitos e fórmulas

**Imputação pela mediana:** o valor ausente é substituído pela mediana da coluna no treino. A mediana é preferida à média porque é **robusta** a valores extremos (em `Fare`, a média 32 é puxada por poucos bilhetes caros; a mediana é 14,45).

**Imputação pela moda:** para categóricos não existe média; usa-se a categoria mais frequente.

**Padronização:**

$$z = \frac{x - \mu_{treino}}{\sigma_{treino}}$$

Como o `StandardScaler` vem **depois** do `SimpleImputer`, $\mu$ e $\sigma$ de `Age` são calculados sobre os dados **já imputados**.

**One-hot encoding:** cada categoria vira uma coluna binária. Para `Embarked`:

| Embarked | Embarked_C | Embarked_Q | Embarked_S |
|---|---|---|---|
| C | 1 | 0 | 0 |
| Q | 0 | 1 | 0 |
| S | 0 | 0 | 1 |

Codificar `C=0, Q=1, S=2` criaria uma **ordem falsa** (como se S fosse "maior" que C), o que distorceria modelos lineares. O *one-hot* evita isso.

**Por que `Pclass` como numérico:** as classes têm ordem natural (1ª < 2ª < 3ª), e a Etapa 2 mostrou a taxa de sobrevivência caindo de forma monótona (62,96% → 47,28% → 24,24%). Tratá-la como número preserva essa ordem com uma única coluna. A alternativa, *one-hot* com 3 colunas, também seria válida.

**Por que dentro do Pipeline (sem vazamento):** imputação e padronização **aprendem** valores dos dados. Dentro do Pipeline, na validação cruzada (Etapa 6), esses valores são reaprendidos **em cada dobra só com as dobras de treino**, e no teste (Etapa 9) são usados os valores do treino completo.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| `ColumnTransformer` | Ferramenta padrão do scikit-learn para tratar colunas de tipos diferentes num único objeto, compatível com Pipeline |
| Mediana para numéricos | Robusta a extremos; `Age` e `Fare` são assimétricas |
| Moda para categóricos | Única estatística de "centro" para texto; só 2 ausentes em `Embarked` |
| `StandardScaler` para todos os numéricos | Necessário para o SGD (sensível à escala); inofensivo para a RF |
| `OneHotEncoder` | Sem ordem artificial entre categorias |
| `handle_unknown='ignore'` | Robustez: na validação cruzada ou no teste pode aparecer uma categoria ausente no treino daquela divisão |
| `Pclass` numérico ordinal | Simples, preserva a ordem e coerente com a EDA |
| Inspecionar com `clone` | Mostra o que é aprendido sem "contaminar" o objeto que vai para os modelos |
| Não transformar `X_test` aqui | O teste fica reservado; no Pipeline, a transformação do teste acontece automaticamente na Etapa 9 |
| Sem engenharia de atributos | Decisão do plano (simplicidade); citada como extensão |

---

## 6. Interpretação dos resultados obtidos

**Configuração** (representação exibida no notebook): `ColumnTransformer` com `num` (`SimpleImputer(strategy='median')` → `StandardScaler()`) para `['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']` e `cat` (`SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`) para `['Sex', 'Embarked']`.

**Valores aprendidos no treino (numéricos):**

| Atributo | Mediana (imputação) | Média (padronização) | Desvio (padronização) |
|---|---|---|---|
| Pclass | 3,000 | 2,309 | 0,833 |
| Age | **28,500** | 29,556 | 13,016 |
| SibSp | 0,000 | 0,493 | 1,060 |
| Parch | 0,000 | 0,390 | 0,838 |
| Fare | 14,454 | 31,820 | 48,025 |

- A mediana de `Age` no treino (**28,5**) é diferente da do conjunto completo (28,0, Etapa 2). Isso mostra que o pré-processador aprendeu **só com o treino**.
- O desvio de `Age` (13,016) é menor que o da Etapa 2 (14,53), porque os 137 valores imputados são todos iguais à mediana, o que reduz a dispersão. É um efeito conhecido da imputação por um valor único.
- O desvio de `Fare` (48,0) é maior que a média (31,8), confirmando a forte assimetria.

**Categóricos:**

```
Sex: moda (imputação) = 'male'; categorias = ['female', 'male']
Embarked: moda (imputação) = 'S'; categorias = ['C', 'Q', 'S']
```

**Transformação do treino:**

```
Tipo do resultado: ndarray
Formato: (712, 7) -> (712, 10)
Valores ausentes após a transformação: 0
Colunas geradas (10): num__Pclass, num__Age, num__SibSp, num__Parch, num__Fare,
                      cat__Sex_female, cat__Sex_male, cat__Embarked_C, cat__Embarked_Q, cat__Embarked_S
```

Os 7 atributos viraram **10 colunas**: 5 numéricas e 2 + 3 colunas *one-hot*. Não resta nenhum ausente, e o resultado é uma matriz NumPy densa.

**Exemplo** (primeiro passageiro do treino com idade ausente, índice original 692):

| | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |
|---|---|---|---|---|---|---|---|
| Antes | 3 | male | NaN | 0 | 0 | 56,4958 | S |

| num__Pclass | num__Age | num__SibSp | num__Parch | num__Fare | Sex_female | Sex_male | Emb_C | Emb_Q | Emb_S |
|---|---|---|---|---|---|---|---|---|---|
| 0,830 | −0,081 | −0,465 | −0,466 | 0,514 | 0 | 1 | 0 | 0 | 1 |

Conferência manual:
- `Age`: ausente → 28,5 (mediana) → $(28{,}5 - 29{,}556)/13{,}016 \approx -0{,}081$;
- `Pclass`: $(3 - 2{,}309)/0{,}833 \approx 0{,}83$;
- `Sex = male` → `[0, 1]`; `Embarked = S` → `[0, 0, 1]`.

---

## 7. Erros comuns e limitações

- **Ajustar o pré-processador em todos os dados** (`fit(X)` antes da divisão): mediana, média e desvio incluiriam o teste. É vazamento.
- **Usar `fit_transform` no teste:** o teste deve só passar por `transform` com os valores do treino. No Pipeline, isso é automático.
- **Codificar categorias como 0, 1, 2:** cria uma ordem artificial entre portos ou sexos.
- **Esquecer `handle_unknown='ignore'`:** uma categoria rara ausente numa dobra de treino geraria erro em outra.
- **Confundir a média da padronização com a média original:** a de `Age` (29,556) é calculada após a imputação.
- **Limitação, imputação simples:** todos os 137 passageiros sem idade recebem 28,5, o que reduz a variância e ignora relações (por exemplo, idade × classe). Imputações mais elaboradas ficam como extensão.
- **Limitação, `Pclass` ordinal:** supõe que a "distância" entre 1ª e 2ª classe é igual à entre 2ª e 3ª. É uma simplificação.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que usar `ColumnTransformer`?** Porque colunas numéricas e categóricas precisam de tratamentos diferentes, e ele aplica cada um às suas colunas num único objeto compatível com Pipeline.
2. **Por que mediana e não média para `Age`?** A mediana é robusta a valores extremos.
3. **Por que *one-hot* e não números 0/1/2 para `Embarked`?** Números criariam uma ordem que não existe entre os portos.
4. **Por que 10 colunas, se havia 7 atributos?** `Sex` virou 2 colunas e `Embarked` virou 3; os 5 numéricos continuam 1 coluna cada.
5. **O pré-processamento usou o teste?** Não. A cópia de inspeção foi ajustada só com `X_train`; nos modelos, ele fica dentro do Pipeline e é ajustado só no treino de cada divisão.
6. **Por que a mediana de `Age` é 28,5 aqui e 28,0 na Etapa 2?** Na Etapa 2 foram usados os 891 passageiros; aqui, só os 712 do treino.
7. **O que acontece com uma categoria nova no teste?** Com `handle_unknown='ignore'`, todas as colunas daquele atributo ficam 0, sem erro.
8. **A Random Forest precisa da padronização?** Não, mas ela não atrapalha. Usar o mesmo pré-processador simplifica o código.

---

## 9. Como apresentar oralmente

> "Os modelos só aceitam números e não aceitam valores ausentes, mas o Titanic tem idade faltando, porto de embarque faltando e colunas de texto. Montei um `ColumnTransformer` com dois tratamentos. Nos numéricos, preencho os ausentes com a mediana e padronizo. Nos categóricos, sexo e porto, preencho com o valor mais frequente e transformo cada categoria numa coluna de zeros e uns. Ajustei uma cópia só no treino para mostrar o que é aprendido: a mediana da idade é 28,5 e o porto mais comum é Southampton. Os 7 atributos viraram 10 colunas numéricas, sem nenhum ausente. No exemplo, um passageiro sem idade recebeu 28,5, que padronizado vale −0,08. Esse pré-processador vai dentro do pipeline de cada modelo, para que tudo seja aprendido só com os dados de treino."

Mostrar na tela: a célula do `ColumnTransformer`, a tabela de valores aprendidos e o exemplo antes × depois.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1, NumPy 2.5.4 e pandas 3.0.6. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Transformação de colunas**

| Item | Para conferir | Link |
|---|---|---|
| `ColumnTransformer` | Formato `(nome, transformador, colunas)`, `named_transformers_`, `get_feature_names_out` | [scikit-learn — ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) |
| Guia do `ColumnTransformer` | Dados heterogêneos | [scikit-learn — ColumnTransformer for heterogeneous data](https://scikit-learn.org/stable/modules/compose.html#column-transformer) |
| Exemplo oficial | Pipeline com dados mistos (usa justamente o Titanic) | [scikit-learn — Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) |
| `get_feature_names_out` | Nomes das colunas geradas | [scikit-learn — Glossary: get_feature_names_out](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out) |

**Imputação, codificação e padronização**

| Item | Para conferir | Link |
|---|---|---|
| `SimpleImputer` | `strategy` (`median`, `most_frequent`) e `statistics_` | [scikit-learn — SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) |
| Guia de imputação | Estratégias para valores ausentes | [scikit-learn — Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) |
| `OneHotEncoder` | `handle_unknown`, `categories_` | [scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) |
| Atributos categóricos | Por que e como codificar | [scikit-learn — Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features) |
| `StandardScaler` | `mean_`, `scale_` | [scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) |

**Pipeline e boas práticas**

| Item | Para conferir | Link |
|---|---|---|
| `make_pipeline` | Encadear transformações (`named_steps`) | [scikit-learn — make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) |
| `clone` | Cópia não ajustada | [scikit-learn — clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) |
| Vazamento de dados | Por que transformar dentro do Pipeline | [scikit-learn — Common pitfalls: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |

**NumPy e pandas**

| Item | Para conferir | Link |
|---|---|---|
| `argmax` | Primeira posição `True` de uma máscara | [NumPy — numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) |
| `iloc` | Selecionar a linha por posição | [pandas — DataFrame.iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html) |
| `pd.isnull` (`pd.isna`) | Contar ausentes no resultado | [pandas — isna](https://pandas.pydata.org/docs/reference/api/pandas.isna.html) |
