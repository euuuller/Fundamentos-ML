# Etapa 2 — Inspeção inicial, definição do alvo e análise exploratória simples

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seções **2.1. Definição do Alvo** e **2.2. Descrição dos Dados**.
Correspondência na referência: seções 2.1 e 2.2 da parte MNIST de `Docs/reference/1-Arquivo.ipynb` (conferência de nulos, distribuição das classes em barras com %), adaptadas a dados tabulares.

---

## 1. Objetivo e relação com o exercício

O enunciado define **Sobreviveu = 1** e **Não Sobreviveu = 0**. Esta etapa:

1. **confere** os valores reais de `Survived` e corrige o tipo (`float` → `int`);
2. mede o **desbalanceamento** das classes;
3. identifica **tipos** (numérico × texto) e **valores ausentes**, que definem o pré-processamento da Etapa 4;
4. descreve **associações** entre atributos e sobrevivência (sexo, classe, idade), sem afirmar causas;
5. **define os atributos** de entrada e justifica os descartes.

---

## 2. Código implementado, por blocos

### Bloco 1 — Conferência e conversão do alvo

```python
print(data['Survived'].value_counts(dropna=False).sort_index())
print(f"Tipo original: {data['Survived'].dtype}")
data['Survived'] = data['Survived'].astype(int)
```

`value_counts(dropna=False)` mostra todos os valores, incluindo ausentes, se existissem. Como só há 0.0 e 1.0, a conversão com `astype(int)` é segura. **Não** há inversão: ao contrário do Exercício 01, a codificação original já é a do enunciado.

### Bloco 2 — Distribuição das classes (tabela e gráfico)

```python
nomes_classes = {0: 'Não Sobreviveu (0)', 1: 'Sobreviveu (1)'}
serie_y = data['Survived'].map(nomes_classes)
distribuicao = pd.DataFrame({'Quantidade': ..., 'Porcentagem': ...})
```

Mesmo padrão da referência e do Exercício 01: quantidade e porcentagem lado a lado e gráfico de barras em %.

### Bloco 3 — `info()`, ausentes e estatísticas

```python
data.info()
ausentes = pd.DataFrame({'Ausentes': data.isnull().sum(),
                         'Porcentagem': (data.isnull().mean()).map('{:.2%}'.format)})
ausentes[ausentes['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].describe().T.round(2)
```

- `info()`: tipo e quantidade de valores não nulos por coluna.
- `isnull().mean()`: a média de valores `True`/`False` é a **proporção** de ausentes.
- `describe()`: estatísticas dos atributos numéricos que serão usados.

### Bloco 4 — Taxa de sobrevivência por grupo

```python
resumo = data.groupby(atributo)['Survived'].agg(Passageiros='count', Taxa_sobrevivencia='mean')
```

Como `Survived` vale 0 ou 1, a **média** em cada grupo é a **taxa de sobrevivência** do grupo. A agregação nomeada (`Passageiros=`, `Taxa_sobrevivencia=`) dá nomes legíveis às colunas. O gráfico mostra as taxas por `Sex` e por `Pclass`, com a **taxa geral** como linha pontilhada de referência.

### Bloco 5 — Histograma da idade por classe

```python
faixas_idade = range(0, 85, 5)
for classe, cor in [(0, 'salmon'), (1, 'skyblue')]:
    idades = data.loc[data['Survived'] == classe, 'Age'].dropna()
    plt.hist(idades, bins=faixas_idade, alpha=0.6, ...)
```

As **mesmas faixas de 5 anos** são usadas nas duas classes. Sem isso, cada histograma calcularia suas próprias faixas, e a comparação visual seria enganosa (ver "Correções" abaixo). O `dropna()` usa só os passageiros com idade conhecida.

### Bloco 6 — Escolha dos atributos

```python
candidatas_descarte = ['PassengerId', 'Name', 'Ticket', 'Cabin']
pd.DataFrame({'Valores distintos': data[candidatas_descarte].nunique(), 'Ausentes (%)': ...})
atributos = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
```

A tabela fundamenta os descartes com números: quantos valores distintos e quantos ausentes cada coluna tem.

### Bloco 7 — Observações (Markdown)

Resume os resultados no notebook.

**Correções feitas durante a etapa** (após conferir os gráficos):
1. a legenda das barras mostrava "Survived" (o nome da coluna) e passou a mostrar "Taxa do grupo";
2. o histograma da idade usava faixas diferentes para cada classe e passou a usar as mesmas faixas de 5 anos.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `Series.value_counts(dropna=False)` | Conta cada valor, incluindo ausentes |
| `Series.astype(int)` | Converte o alvo de `float64` para `int64` |
| `nomes_classes` | `{0: 'Não Sobreviveu (0)', 1: 'Sobreviveu (1)'}`, usado em tabelas e gráficos (e nas próximas etapas) |
| `DataFrame.info()` | Tipos e não nulos por coluna |
| `isnull().sum()` / `isnull().mean()` | Quantidade / proporção de ausentes |
| `describe()` | Estatísticas descritivas |
| `groupby(...)['Survived'].agg(...)` | Contagem e taxa de sobrevivência por grupo |
| `plt.hist(..., bins=faixas_idade)` | Histograma com faixas fixas |
| `ax.axhline(...)` | Linha horizontal da taxa geral |
| `DataFrame.nunique()` | Quantidade de valores distintos por coluna |
| `atributos` | **Lista oficial de atributos de entrada**, usada a partir da Etapa 3 |
| `candidatas_descarte` | Colunas descartadas |

---

## 4. Conceitos

- **Classe positiva:** as métricas do scikit-learn usam `pos_label=1`. Aqui, 1 = Sobreviveu: recall é a fração de sobreviventes reais que o modelo identifica.
- **Desbalanceamento moderado:** 38,38% de positivos. Um modelo que sempre responde "não sobreviveu" acertaria ≈61,6% sem aprender nada.
- **Tipos de atributos:**
  - **numéricos contínuos:** `Age`, `Fare`;
  - **numéricos discretos:** `SibSp`, `Parch`;
  - **ordinal:** `Pclass` (1 < 2 < 3);
  - **categóricos nominais:** `Sex`, `Embarked` (sem ordem; precisam de codificação, Etapa 4).
- **Taxa de sobrevivência por grupo:**

  $$\text{taxa}_g = \frac{\text{sobreviventes do grupo } g}{\text{passageiros do grupo } g} = \overline{\text{Survived}}_g$$

- **Associação ≠ causalidade:** uma taxa diferente entre grupos mostra que as variáveis andam juntas nos dados. Os grupos podem diferir em outros aspectos (por exemplo, classe e sexo estão misturados), e os dados não explicam *por que* a taxa difere.
- **Cardinalidade:** colunas com um valor distinto por passageiro (`PassengerId`, `Name`) funcionam como identificadores: o modelo poderia "decorar" passageiros, mas não generalizar para novos.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Conferir os valores antes de converter | A especificação exige confirmar a codificação do alvo |
| `astype(int)` sem inversão | Os valores já são 0 = não sobreviveu e 1 = sobreviveu, como no enunciado |
| Mostrar ausentes em quantidade e % | Define o tratamento: imputar `Age` (≈20%) e `Embarked` (2 casos); descartar `Cabin` (77%) |
| Taxas por `Sex` e `Pclass` | Atributos categóricos com poucos grupos, fáceis de comparar e de explicar |
| Histograma de `Age` com faixas iguais | Comparação justa entre as classes |
| Linha da taxa geral nos gráficos | Mostra quais grupos estão acima ou abaixo da média |
| Descartar `PassengerId` e `Name` | 891 valores distintos (identificadores); extrair o título do nome seria engenharia de atributos, fora do escopo definido |
| Descartar `Ticket` | 681 valores distintos de texto livre, sem estrutura simples |
| Descartar `Cabin` | 77,10% ausente; imputar tanta informação seria inventar dados |
| Manter `Age` mesmo com 19,87% ausente | É uma informação potencialmente útil; a imputação pela mediana será feita no Pipeline (Etapa 4) |
| EDA no conjunto completo | É só descritiva; nenhum parâmetro do modelo é aprendido aqui. A imputação (que aprende mediana e moda) será ajustada só no treino |

---

## 6. Interpretação dos resultados obtidos

**Alvo:**

```
Survived
0.0    549
1.0    342
Tipo original: float64
Tipo após a conversão: int64
```

| Classe | Quantidade | Porcentagem |
|---|---|---|
| Não Sobreviveu (0) | 549 | 61,62% |
| Sobreviveu (1) | 342 | 38,38% |

A codificação já está correta (1 = Sobreviveu), e o tipo foi corrigido. O desbalanceamento (38% × 62%) é parecido com o do Exercício 01 (37% × 63%).

**Tipos (`info()`):** 891 linhas; `Survived`, `Pclass`, `SibSp`, `Parch` inteiros; `Age`, `Fare` reais; `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` texto (`str`).

**Ausentes:**

| Coluna | Ausentes | % |
|---|---|---|
| Cabin | 687 | 77,10% |
| Age | 177 | 19,87% |
| Embarked | 2 | 0,22% |

**Estatísticas (atributos numéricos):**

| Atributo | n | média | mediana | mín | máx |
|---|---|---|---|---|---|
| Pclass | 891 | 2,31 | 3,00 | 1 | 3 |
| Age | 714 | 29,70 | 28,00 | 0,42 | 80,00 |
| SibSp | 891 | 0,52 | 0,00 | 0 | 8 |
| Parch | 891 | 0,38 | 0,00 | 0 | 6 |
| Fare | 891 | 32,20 | 14,45 | 0,00 | 512,33 |

- `Fare` é muito assimétrica: média de 32,20 contra mediana de 14,45, e máximo de 512,33. Poucos bilhetes caros puxam a média para cima.
- A mediana de `Pclass` é 3: mais da metade dos passageiros estava na 3ª classe.
- As escalas são diferentes (`Fare` até 512; `Parch` até 6), o que reforça a padronização para o SGD.

**Taxas de sobrevivência por grupo:**

| Grupo | Passageiros | Taxa |
|---|---|---|
| Sex = female | 314 | **74,20%** |
| Sex = male | 577 | **18,89%** |
| Pclass = 1 | 216 | 62,96% |
| Pclass = 2 | 184 | 47,28% |
| Pclass = 3 | 491 | 24,24% |
| Geral | 891 | 38,38% |

Nos dados, sexo e classe estão fortemente **associados** à sobrevivência: a taxa entre mulheres é cerca de 4 vezes a dos homens, e a da 1ª classe é cerca de 2,6 vezes a da 3ª. Isso indica que esses atributos devem ser informativos para os modelos. **Não** se pode concluir daqui a causa das diferenças, e os dois fatores se sobrepõem (a composição por sexo pode variar entre as classes).

**Histograma da idade** (faixas de 5 anos, só idades conhecidas):
- na faixa de **0 a 5 anos**, há mais sobreviventes que não sobreviventes;
- nas faixas de **15 a 35 anos**, predominam claramente os não sobreviventes;
- nas demais faixas, as distribuições se sobrepõem bastante: a idade sozinha separa pouco as classes.

*Verificação auxiliar (fora do notebook, só para dimensionar o efeito):* entre os 714 passageiros com idade conhecida, os 62 menores de 10 anos tiveram taxa de sobrevivência de 61,29%, contra 40,62% no total com idade conhecida.

**Atributos:**

| Coluna | Valores distintos | Ausentes |
|---|---|---|
| PassengerId | 891 | 0,00% |
| Name | 891 | 0,00% |
| Ticket | 681 | 0,00% |
| Cabin | 147 | 77,10% |

Atributos de entrada (7): `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`.

---

## 7. Erros comuns e limitações

- **Inverter o alvo por hábito:** no Exercício 01 foi preciso inverter; aqui **não**. Conferir sempre, nunca assumir.
- **Afirmar causalidade** ("ser mulher fez sobreviver"): os dados mostram associação; há fatores misturados e não observados.
- **Comparar histogramas com faixas diferentes:** produz impressões visuais falsas. Corrigido nesta etapa.
- **Imputar antes de dividir treino e teste:** a mediana de `Age` usaria informação do teste (vazamento). A imputação será feita no Pipeline, só com o treino.
- **Usar `Name` ou `PassengerId` como atributo:** são identificadores, e o modelo os memorizaria sem generalizar.
- **Limitação:** a EDA olha um atributo de cada vez. Interações (como sexo × classe) não foram analisadas, para manter a etapa simples.
- **Limitação:** o descarte de `Cabin`, `Ticket` e `Name` pode desperdiçar algum sinal (por exemplo, o título no nome). Isso fica como possível extensão, fora do escopo definido.

---

## 8. Perguntas que o professor poderia fazer

1. **Precisou inverter o alvo, como no Exercício 01?** Não. `Survived` já é 1 = Sobreviveu; só foi preciso converter de `float` para `int`.
2. **Por que `Survived` era `float`?** O `concat` com o `test.csv` introduziu `NaN`, e inteiros não representam ausentes no pandas.
3. **As classes são desbalanceadas?** Moderadamente: 38,38% de sobreviventes. Um modelo que sempre diz "não sobreviveu" teria ≈61,6% de acurácia.
4. **Por que descartar `Cabin`?** Tem 77,10% de ausentes; preencher isso seria criar dados.
5. **E `Age`, que também tem ausentes?** São 19,87%; o atributo é útil e será imputado pela mediana, aprendida só no treino.
6. **O que significa a taxa de 74,20% para mulheres?** Que, nos dados, 74,20% das passageiras sobreviveram, contra 18,89% dos homens. É uma associação, não uma prova de causa.
7. **Por que descartar `Name`?** Cada passageiro tem um nome único (891 valores distintos), então não generaliza. Extrair o título seria engenharia de atributos, fora do escopo.

---

## 9. Como apresentar oralmente

> "Primeiro conferi o alvo: `Survived` só tem 0 e 1, já na codificação do enunciado, então não precisei inverter como no exercício anterior; só corrigi o tipo para inteiro. São 549 não sobreviventes e 342 sobreviventes, cerca de 38%. Depois olhei os ausentes: a cabine falta em 77% dos passageiros e foi descartada; a idade falta em 20% e será preenchida pela mediana dentro do pipeline; o porto de embarque falta em só 2 casos. Nos dados, a taxa de sobrevivência foi de 74% entre mulheres e 19% entre homens, e cai de 63% na primeira classe para 24% na terceira. São associações fortes, que indicam atributos úteis, mas não provam causa. Por fim, escolhi 7 atributos e descartei identificadores e texto livre: o ID, o nome, o bilhete e a cabine."

Mostrar na tela: a conferência do alvo, a tabela de ausentes, o gráfico de taxas por sexo e classe e a tabela de descartes.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa pandas 3.0.6 e matplotlib 3.11.2. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**pandas**

| Item | Para conferir | Link |
|---|---|---|
| `Series.value_counts` | `dropna` e `normalize` | [pandas — Series.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) |
| `Series.astype` | Conversão de tipo | [pandas — Series.astype](https://pandas.pydata.org/docs/reference/api/pandas.Series.astype.html) |
| Tipos de dados | `int64`, `float64`, `str` | [pandas — dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes) |
| `Series.map` | Rótulos legíveis | [pandas — Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) |
| `DataFrame.info` | Resumo de tipos e não nulos | [pandas — DataFrame.info](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) |
| `isnull` / `mean` | Contagem e proporção de ausentes | [pandas — DataFrame.isnull](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) · [DataFrame.mean](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html) |
| Dados ausentes | Como o pandas representa e trata ausentes | [pandas — Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) |
| `describe` | Estatísticas descritivas | [pandas — DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) |
| `groupby` | Agrupar e agregar | [pandas — DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) |
| Agregação nomeada | `agg(Nome='funcao')` | [pandas — Named aggregation](https://pandas.pydata.org/docs/user_guide/groupby.html#named-aggregation) |
| `nunique` | Valores distintos por coluna | [pandas — DataFrame.nunique](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html) |

**Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `plt.hist` | `bins` como sequência de limites | [Matplotlib — pyplot.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) |
| `ax.axhline` | Linha da taxa geral | [Matplotlib — Axes.axhline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html) |
| `plt.subplots` | Dois gráficos lado a lado | [Matplotlib — pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) |
| `PercentFormatter` | Eixo em porcentagem | [Matplotlib — ticker.PercentFormatter](https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.PercentFormatter) |

**Dados**

| Item | Para conferir | Link |
|---|---|---|
| Dicionário oficial | Significado de cada coluna | [Kaggle — Titanic: Data](https://www.kaggle.com/competitions/titanic/data) |
