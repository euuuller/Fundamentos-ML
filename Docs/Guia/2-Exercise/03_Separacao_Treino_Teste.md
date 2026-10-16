# Etapa 3 — Separação dos dados de treino e teste

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seção **2.3. Separação dos Dados de Treino e Teste**.
Correspondência na referência: seção 2.3 de `Docs/reference/1-Arquivo.ipynb` (mesma função e mesmos parâmetros). É a mesma estratégia da Etapa 3 do Exercício 01.

---

## 1. Objetivo e relação com o exercício

Para saber como os modelos se comportam com **passageiros que nunca viram**, uma parte dos dados é separada **antes de qualquer pré-processamento ou treino** e fica guardada até a avaliação final (Etapa 9).

- **Treino (80%):** pré-processamento (Etapa 4), treino (Etapa 5), validação cruzada (Etapa 6), estudo do limiar e curvas ROC (Etapa 7), matrizes fora da amostra (Etapa 8).
- **Teste (20%):** usado **uma única vez**, na Etapa 9.

Este teste é **próprio**: sai dos 891 passageiros rotulados. O `test.csv` do Kaggle não serve, porque não tem `Survived` (Etapa 1).

---

## 2. Código implementado, por blocos

### Bloco 1 — Entradas, saída e divisão estratificada

```python
from sklearn.model_selection import train_test_split

X = data[atributos]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

- `X` recebe **somente os 7 atributos** escolhidos na Etapa 2. As colunas descartadas (`PassengerId`, `Name`, `Ticket`, `Cabin`) e o alvo ficam de fora.
- `X` e `y` são objetos do **pandas** (`DataFrame` e `Series`), não arrays NumPy como no Exercício 01. Isso é importante: o `ColumnTransformer` da Etapa 4 vai selecionar colunas **pelo nome**.
- A chamada do `train_test_split` é igual à da referência e à do Exercício 01.

### Bloco 2 — Conferência das proporções

```python
comparacao_split = pd.DataFrame({
    'Completo': y.map(nomes_classes).value_counts(),
    'Treino': y_train.map(nomes_classes).value_counts(),
    'Teste': y_test.map(nomes_classes).value_counts()
})
print((comparacao_split / comparacao_split.sum()).map('{:.2%}'.format))
```

Mesmo padrão do Exercício 01: contagens por classe nos três conjuntos e proporções por coluna (cada coluna dividida pelo seu total).

### Bloco 3 — Ausentes em cada parte

```python
pd.DataFrame({
    'Treino': X_train[['Age', 'Embarked']].isnull().sum(),
    'Teste': X_test[['Age', 'Embarked']].isnull().sum()
})
```

Mostra como os ausentes se dividiram. A Etapa 4 vai **aprender** a mediana de `Age` e a moda de `Embarked` **só com o treino** e aplicá-las aos ausentes do teste.

### Bloco 4 — Observações (Markdown)

Resume os números obtidos.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `X = data[atributos]` | `DataFrame` 891 × 7 com as entradas (seleção de colunas por nome) |
| `y = data['Survived']` | `Series` com o alvo (0/1, inteiro) |
| `train_test_split(X, y, ...)` | Embaralha e divide; devolve `X_train, X_test, y_train, y_test`, nessa ordem |
| `test_size=0.2` | 20% para teste: 891 × 0,2 = 178,2, arredondado **para cima**, ou seja, 179 |
| `stratify=y` | Mantém a proporção de classes em cada parte |
| `random_state=42` | Divisão reproduzível |
| `X_train`, `y_train` | 712 passageiros: tudo o que as Etapas 4 a 8 podem usar |
| `X_test`, `y_test` | 179 passageiros: **reservados para a Etapa 9** |
| `comparacao_split` | Contagens por classe e conjunto |

---

## 4. Conceitos

- **Hold-out:** um conjunto de teste que não participa de nenhuma decisão. Se participasse, a métrica final seria otimista.
- **Estratificação:** a divisão é feita dentro de cada classe, mantendo a proporção de sobreviventes (≈38%) nas duas partes.
- **Vazamento no pré-processamento:** imputação e padronização **aprendem** valores (mediana, moda, média, desvio). Se aprendessem com todos os 891 passageiros, o teste influenciaria o treino. Por isso a divisão vem **antes** da Etapa 4.
- **Sensibilidade das métricas no teste:**

  $$\Delta Recall = \frac{1}{69} \approx 1{,}45 \text{ p.p. por sobrevivente} \qquad \Delta Especificidade = \frac{1}{110} \approx 0{,}91 \text{ p.p. por não sobrevivente}$$

  O teste do Titanic é maior que o do Exercício 01 (179 × 114), mas ainda é pequeno: diferenças de 1 ou 2 passageiros não devem ser tratadas como conclusivas.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| 80/20, estratificado, `random_state=42` | Mesmos parâmetros da referência e do Exercício 01 |
| `stratify=y` | Classes moderadamente desbalanceadas (38%); a proporção de sobreviventes afeta diretamente recall, precisão e F1 |
| Selecionar `X = data[atributos]` antes de dividir | As colunas descartadas não chegam aos modelos em nenhum momento |
| Manter `X` como `DataFrame` | Permite ao `ColumnTransformer` escolher colunas por nome (`'Age'`, `'Sex'`...) |
| Dividir **antes** de imputar, codificar e padronizar | Evita vazamento: tudo o que é aprendido nas transformações vem só do treino |
| Mostrar os ausentes por parte | Deixa claro que o teste também tem ausentes (40 em `Age`) e que eles serão preenchidos com valores aprendidos no treino |
| Um conjunto de teste próprio, não o `test.csv` | O `test.csv` não tem rótulo |

---

## 6. Interpretação dos resultados obtidos

**Formatos:**

```
Formato do conjunto de treino: (712, 7)
Formato do conjunto de teste:  (179, 7)
```

712 + 179 = 891, ou seja, nenhum passageiro perdido ou duplicado, e os 7 atributos estão nas duas partes.

**Quantidade e proporção por classe:**

| Classe | Completo | Treino | Teste |
|---|---|---|---|
| Não Sobreviveu (0) | 549 (61,62%) | 439 (61,66%) | 110 (61,45%) |
| Sobreviveu (1) | 342 (38,38%) | 273 (38,34%) | 69 (38,55%) |

A estratificação funcionou: a proporção de sobreviventes varia menos de 0,25 ponto percentual entre os conjuntos.

**Ausentes por parte:**

| Atributo | Treino | Teste |
|---|---|---|
| Age | 137 | 40 |
| Embarked | 2 | 0 |

Os 177 ausentes de `Age` se dividiram em 137 + 40, aproximadamente 80/20. Os 2 ausentes de `Embarked` ficaram no treino, mas isso não muda a regra: a moda é aprendida no treino e, se um passageiro novo tiver `Embarked` ausente, recebe essa moda.

**Leitura:** o treino tem **273 sobreviventes** para aprender e validar, e o teste tem **69**. No teste, cada sobrevivente não identificado custa ≈1,45 p.p. de recall.

---

## 7. Erros comuns e limitações

- **Imputar ou padronizar antes de dividir:** a mediana de `Age` e as médias usadas na padronização incluiriam o teste, o que é vazamento.
- **Esquecer de selecionar os atributos:** passar `data` inteiro incluiria `Name`, `Ticket` e até o alvo (`Survived`) como entrada.
- **Inverter a ordem do retorno** do `train_test_split`.
- **Converter `X` para array NumPy agora:** perderia os nomes das colunas usados pelo `ColumnTransformer`.
- **Limitação:** um único *split* de 179 passageiros. Outra semente daria números um pouco diferentes, e por isso a validação cruzada no treino (Etapa 6) é a base da comparação entre modelos.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que não usar o `test.csv` como teste?** Ele não tem a coluna `Survived`, então não há como calcular métricas.
2. **Por que dividir antes do pré-processamento?** Porque imputação e padronização aprendem valores dos dados; se vissem o teste, haveria vazamento.
3. **O que a estratificação garantiu?** A mesma proporção de sobreviventes: 38,34% no treino e 38,55% no teste, contra 38,38% no total.
4. **Por que 179 e não 178 passageiros no teste?** 20% de 891 = 178,2; o `train_test_split` arredonda o teste para cima.
5. **O teste tem valores ausentes. Como serão tratados?** Com a mediana (`Age`) e a moda (`Embarked`) aprendidas **no treino**, dentro do Pipeline.
6. **Por que `X` continua sendo um `DataFrame`?** Para o `ColumnTransformer` selecionar as colunas numéricas e categóricas pelo nome.

---

## 9. Como apresentar oralmente

> "Antes de qualquer transformação, separei 20% dos 891 passageiros rotulados como teste, com o mesmo `train_test_split` estratificado da referência. Esse teste fica guardado até a avaliação final. Usei só os 7 atributos escolhidos. A estratificação manteve a proporção de sobreviventes: 38,3% no treino e 38,6% no teste. Ficaram 712 passageiros para treino e 179 para teste, com 69 sobreviventes. Dividir antes do pré-processamento é essencial aqui, porque a idade tem valores ausentes: a mediana usada para preenchê-los vai ser aprendida só no treino, inclusive para os 40 ausentes do teste."

Mostrar na tela: a célula do `train_test_split`, a tabela de proporções e a tabela de ausentes por parte.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1 e pandas 3.0.6. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**scikit-learn**

| Item | Para conferir | Link |
|---|---|---|
| `train_test_split` | `test_size`, `stratify`, `random_state` e a ordem do retorno | [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| Estratificação | Divisões que preservam a proporção de classes | [scikit-learn — Cross-validation iterators with stratification](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) |
| `random_state` | O que o parâmetro controla | [scikit-learn — Glossary: random_state](https://scikit-learn.org/stable/glossary.html#term-random_state) |
| Vazamento de dados | Por que dividir antes de transformar | [scikit-learn — Common pitfalls: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |
| Pré-processamento inconsistente | Por que treino e teste devem passar pela mesma transformação aprendida no treino | [scikit-learn — Inconsistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing) |

**pandas**

| Item | Para conferir | Link |
|---|---|---|
| Seleção de colunas | `data[atributos]` com lista de nomes | [pandas — Selection by label](https://pandas.pydata.org/docs/user_guide/indexing.html#selection-by-label) |
| `DataFrame` | Construção da tabela de comparação | [pandas — DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) |
| Divisão `DataFrame / Series` | Proporções por coluna | [pandas — DataFrame.div](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.div.html) |
| `DataFrame.map` | Formatação em porcentagem | [pandas — DataFrame.map](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.map.html) |
