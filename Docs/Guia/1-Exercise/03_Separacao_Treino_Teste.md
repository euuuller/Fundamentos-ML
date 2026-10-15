# Etapa 3 — Separação dos dados de treino e teste

Notebook: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`, seção **2.3. Separação dos Dados de Treino e Teste**.
Correspondência na referência: seção 2.3 de `Docs/reference/1-Arquivo.ipynb` (mesma função e mesmos parâmetros).

---

## 1. Objetivo e relação com o exercício

O objetivo do exercício é saber **como os modelos se comportam com tumores que nunca viram**. Para medir isso honestamente, uma parte dos dados é separada **agora**, antes de qualquer treino, e fica guardada até a avaliação final (Etapa 8).

- **Treino (80%):** usado para treinar os modelos, para a validação cruzada (Etapa 5), para estudar o limiar (Etapa 6) e para as matrizes de confusão fora da amostra (Etapa 7).
- **Teste (20%):** usado **uma única vez**, na Etapa 8.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Registra no notebook a regra de que o teste fica reservado e o motivo da estratificação.

### Bloco 2 — Divisão estratificada

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Formato do conjunto de treino: {X_train.shape}")
print(f"Formato do conjunto de teste:  {X_test.shape}")
```

A chamada é idêntica à da referência, só trocando `y_binary` por `y`. A função embaralha as amostras e devolve quatro arrays, **sempre nesta ordem**: `X_train`, `X_test`, `y_train`, `y_test`.

### Bloco 3 — Conferência das proporções

```python
comparacao_split = pd.DataFrame({
    'Completo': pd.Series(y).map(nomes_classes).value_counts(),
    'Treino': pd.Series(y_train).map(nomes_classes).value_counts(),
    'Teste': pd.Series(y_test).map(nomes_classes).value_counts()
})
print(comparacao_split)
print((comparacao_split / comparacao_split.sum()).map('{:.2%}'.format))
```

- Cada coluna conta as classes de um conjunto (com os rótulos legíveis de `nomes_classes`, criados na Etapa 2).
- `comparacao_split.sum()` soma cada coluna (o total de amostras do conjunto). Dividir a tabela por essa soma transforma contagens em **proporções por coluna**.
- `.map('{:.2%}'.format)` formata cada célula como porcentagem.

A referência imprimia treino e teste separadamente; aqui as três distribuições ficam lado a lado, o que facilita conferir a estratificação num único olhar.

### Bloco 4 — Observações (Markdown)

Resume os números obtidos e a sensibilidade do teste pequeno.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `train_test_split(X, y, ...)` | Embaralha e divide os dados em treino e teste |
| `test_size=0.2` | 20% para teste (569 × 0,2 = 113,8; a função arredonda o teste **para cima**, então ficam 114) |
| `stratify=y` | Mantém em cada parte a proporção de classes de `y` |
| `random_state=42` | Fixa o embaralhamento: a mesma divisão em toda execução, o que torna o trabalho reproduzível |
| `X_train`, `y_train` | (455, 30) e (455,): tudo o que os modelos e a validação podem usar |
| `X_test`, `y_test` | (114, 30) e (114,): **reservados para a Etapa 8** |
| `comparacao_split` | Tabela de contagens por classe e por conjunto |

---

## 4. Conceitos

- **Generalização:** o que interessa é o desempenho em dados novos, não nos dados de treino. Um modelo pode decorar o treino (*overfitting*) e falhar em casos novos.
- **Hold-out (conjunto de teste reservado):** separar dados que não participam de **nenhuma** decisão. Se o teste fosse usado para escolher modelo ou limiar, a métrica final ficaria otimista, porque as escolhas teriam sido "ajustadas" a ele.
- **Estratificação:** a divisão é feita **dentro de cada classe**. Com proporção $p$ de malignos no total, cada parte recebe aproximadamente a mesma proporção $p$. Sem ela, por acaso, o teste poderia ficar com poucos malignos, e o recall medido nele seria instável.
- **Sensibilidade do recall no teste:**

  $$\text{Recall} = \frac{TP}{TP + FN}, \qquad TP + FN = \text{nº de malignos no teste} = 42$$

  Cada maligno perdido (FN) reduz o recall em $1/42 \approx 2{,}38$ pontos percentuais. É um conjunto pequeno: diferenças de 1 ou 2 tumores entre modelos não devem ser tratadas como conclusivas.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| `test_size=0.2` (80/20) | Mesma proporção da referência; mantém 455 amostras para treinar e validar, o que é importante num dataset pequeno |
| `stratify=y` | As classes são desbalanceadas (37% malignos, Etapa 2), e a métrica prioritária (recall) depende diretamente do número de malignos em cada parte |
| `random_state=42` | Reprodutibilidade (mesmo valor da referência) |
| Divisão **antes** de qualquer transformação ou treino | Garante que nada aprendido (média e desvio do `StandardScaler`, limiar, escolha de modelo) use informação do teste |
| Um único *split* de teste, e não vários | O teste é o "exame final". A estimativa robusta durante o desenvolvimento virá da **validação cruzada estratificada no treino** (Etapa 5), que reaproveita todo o treino em várias divisões |
| Não usar um conjunto de validação separado | Com 455 amostras, separar mais uma parte fixa reduziria muito o treino; a validação cruzada cumpre esse papel sem desperdiçar dados |

---

## 6. Interpretação dos resultados obtidos

**Formatos:**

```
Formato do conjunto de treino: (455, 30)
Formato do conjunto de teste:  (114, 30)
```

455 + 114 = 569, ou seja, nenhuma amostra se perdeu ou foi duplicada. Os 30 atributos foram mantidos.

**Quantidade por classe:**

| Classe | Completo | Treino | Teste |
|---|---|---|---|
| Benigno (0) | 357 | 285 | 72 |
| Maligno (1) | 212 | 170 | 42 |

**Proporção por classe:**

| Classe | Completo | Treino | Teste |
|---|---|---|---|
| Benigno (0) | 62,74% | 62,64% | 63,16% |
| Maligno (1) | 37,26% | 37,36% | 36,84% |

Leitura:
- A estratificação funcionou: a proporção de malignos varia menos de 0,5 ponto percentual entre os três conjuntos. As pequenas diferenças vêm de as contagens precisarem ser números inteiros.
- O treino tem **170 malignos**, que são os exemplos positivos disponíveis para aprender e validar.
- O teste tem **42 malignos**: o recall final será medido sobre esses 42 casos, e cada um vale ≈2,4 pontos percentuais.

---

## 7. Erros comuns e limitações

- **Inverter a ordem do retorno** (por exemplo, `X_train, y_train, X_test, y_test = ...`). A ordem correta é `X_train, X_test, y_train, y_test`.
- **Esquecer `stratify=y`.** Com dados desbalanceados, a proporção de malignos no teste ficaria ao acaso.
- **Padronizar `X` antes de dividir.** O `StandardScaler` aprenderia a média e o desvio usando o teste, o que é vazamento. Neste trabalho, a padronização ficará dentro de um `Pipeline` ajustado só no treino (Etapa 4).
- **"Espiar" o teste:** avaliar no teste, mudar algo e avaliar de novo. Isso transforma o teste em validação e invalida a estimativa final.
- **Limitação:** com 114 amostras, a métrica de teste tem variância considerável; outro `random_state` poderia dar números um pouco diferentes. Por isso, a validação cruzada (Etapa 5) será a base principal da comparação entre modelos, e o teste confirma o resultado.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que separar o teste antes de tudo?** Para que nenhuma decisão (transformação, modelo, limiar) seja influenciada por ele; só assim a avaliação final é imparcial.
2. **O que a estratificação garante?** Que treino e teste tenham a mesma proporção de malignos do conjunto completo: 37,36% e 36,84%, contra 37,26%.
3. **Por que o teste tem 114 e não 113 amostras?** 20% de 569 é 113,8; o `train_test_split` arredonda o tamanho do teste para cima.
4. **Para que serve `random_state=42`?** Para que o embaralhamento seja sempre o mesmo e os resultados possam ser reproduzidos.
5. **Por que não criar também um conjunto de validação fixo?** Com só 455 amostras de treino, a validação cruzada estratificada aproveita melhor os dados: cada amostra é usada para validar uma vez.
6. **O teste é grande o suficiente?** É pequeno: tem 42 malignos, e cada erro muda o recall em ≈2,4 pontos percentuais. Por isso os resultados de teste devem ser lidos junto com os da validação cruzada.

---

## 9. Como apresentar oralmente

> "Antes de treinar qualquer coisa, separei 20% dos dados como teste, com o mesmo `train_test_split` da referência. O teste fica guardado: só vou usá-lo na avaliação final, para não contaminar nenhuma escolha. Usei `stratify=y` porque as classes são desbalanceadas e o recall depende do número de malignos. A tabela mostra que a proporção de malignos ficou praticamente igual: 37,26% no total, 37,36% no treino e 36,84% no teste. Ficaram 455 amostras para treino, com 170 malignos, e 114 para teste, com 42 malignos. Como o teste é pequeno, cada maligno não detectado custa cerca de 2,4 pontos de recall, então vou usar a validação cruzada no treino como base da comparação."

Mostrar na tela: a célula do `train_test_split` e a tabela de proporções lado a lado.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1 e pandas 3.0.6. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**scikit-learn**

| Item | Para conferir | Link |
|---|---|---|
| `train_test_split()` | Parâmetros `test_size`, `stratify`, `random_state`, `shuffle` e a ordem do retorno | [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| Treino × teste e validação cruzada | Por que reservar um teste e como a validação cruzada evita um conjunto de validação fixo (base da Etapa 5) | [scikit-learn — Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance) |
| Estratificação | Divisões que preservam a proporção das classes | [scikit-learn — Cross-validation iterators with stratification](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) |
| `random_state` | O que o parâmetro controla e como usá-lo | [scikit-learn — Glossary: random_state](https://scikit-learn.org/stable/glossary.html#term-random_state) |
| Vazamento de dados | Por que transformar/ajustar só depois da divisão | [scikit-learn — Common pitfalls: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |
| Controle da aleatoriedade | Boas práticas de reprodutibilidade | [scikit-learn — Common pitfalls: Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness) |

**pandas e Python**

| Item | Para conferir | Link |
|---|---|---|
| `Series.map()` | Troca dos códigos 0/1 pelos rótulos | [pandas — Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) |
| `DataFrame.sum()` | Soma por coluna (total de cada conjunto) | [pandas — DataFrame.sum](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html) |
| Divisão `DataFrame / Series` | Divide cada coluna pelo seu total (alinhamento pelos nomes das colunas) | [pandas — DataFrame.div](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.div.html) |
| `DataFrame.map()` | Aplica a formatação em cada célula | [pandas — DataFrame.map](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.map.html) |
| `'{:.2%}'` | Formato de porcentagem com 2 casas | [Python — Format Specification Mini-Language](https://docs.python.org/3/library/string.html#format-specification-mini-language) |
