# Etapa 3 — Separação dos dados de treino e teste

Notebook: `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`, seção **2.3. Separação dos Dados de Treino e Teste**.
Correspondência na referência: seção 2.3 de `Docs/reference/2-Arquivo.ipynb` (mesma função e mesmos parâmetros: `test_size=0.2`, `stratify=y`, `random_state=42`).

---

## 1. Objetivo e relação com o exercício

Separar, **antes de qualquer treino**, um conjunto de teste que só será usado na avaliação final (Etapa 7). Neste exercício, dois cuidados pesam mais que nos anteriores:

- **embaralhar**, porque o dataset vem em blocos por classe (Etapa 2);
- **estratificar**, porque há 3 classes e poucas amostras: sem isso, o acaso poderia deixar uma cultivar sub-representada no teste.

---

## 2. Código implementado, por blocos

### Bloco 1 — Explicação (Markdown)

Registra que o teste fica reservado e por que a divisão embaralha e estratifica.

### Bloco 2 — Divisão

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Primeiros 15 rótulos do treino: {y_train[:15]}")
```

É a chamada da referência. O `train_test_split` usa `shuffle=True` por padrão, ou seja, embaralha antes de dividir. Imprimir os primeiros rótulos do treino comprova o embaralhamento: eles aparecem misturados, e não em blocos.

### Bloco 3 — Proporções por classe

```python
comparacao_split = pd.DataFrame({
    'Completo': pd.Series(y).map(nomes_classes).value_counts(),
    'Treino': pd.Series(y_train).map(nomes_classes).value_counts(),
    'Teste': pd.Series(y_test).map(nomes_classes).value_counts()
}).sort_index()
```

Mesmo padrão dos exercícios anteriores, agora com 3 classes, com nomes legíveis e ordenado por classe.

### Bloco 4 — Observações (Markdown)

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `train_test_split(X, y, ...)` | Embaralha e divide; retorna `X_train, X_test, y_train, y_test` |
| `test_size=0.2` | 20%: 178 × 0,2 = 35,6, arredondado **para cima** = 36 amostras |
| `stratify=y` | Mesma proporção de cada classe nas duas partes |
| `shuffle=True` (padrão) | Embaralha antes de dividir, essencial com dados ordenados |
| `random_state=42` | Divisão reproduzível |
| `X_train`, `y_train` | 142 amostras: usadas nas Etapas 4 a 6 |
| `X_test`, `y_test` | 36 amostras: **reservadas para a Etapa 7** |
| `comparacao_split` | Contagens por classe e conjunto |

---

## 4. Conceitos

- **Estratificação em multiclasse:** a divisão é feita dentro de cada classe. Com proporções $p_0, p_1, p_2$ no conjunto completo, treino e teste recebem aproximadamente as mesmas proporções.
- **Embaralhamento:** sem ele, `train_test_split` pegaria as últimas 36 linhas para o teste, que no Wine são quase todas `class_2`. A estratificação também evita isso, mas o embaralhamento garante que, **dentro** de cada classe, as amostras escolhidas sejam aleatórias.
- **Peso de cada amostra no teste:**

  $$\Delta Acc = \frac{1}{36} \approx 2{,}8 \text{ p.p.} \qquad \Delta Recall_{c} = \frac{1}{n_c^{teste}}: \quad \frac{1}{12} \approx 8{,}3;\ \frac{1}{14} \approx 7{,}1;\ \frac{1}{10} = 10 \text{ p.p.}$$

  Um único erro numa classe do teste muda o recall dela em 7 a 10 pontos. Métricas por classe no teste serão bastante "granulares".
- **Validação cruzada (adiantando a Etapa 5):** com 142 amostras no treino, cada dobra das 5 terá cerca de 28 amostras. A comparação entre modelos se apoiará nas 5 avaliações, e não só no teste.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| 80/20, `stratify=y`, `random_state=42` | Parâmetros da referência e dos exercícios anteriores |
| Embaralhar (padrão) | O dataset vem ordenado por classe |
| Estratificar | 3 classes, poucas amostras e leve desbalanceamento |
| Mostrar os primeiros rótulos do treino | Evidência simples de que o embaralhamento aconteceu |
| Não usar um conjunto de validação separado | Com 142 amostras, separar mais uma parte fixa reduziria muito o treino; a validação cruzada cumpre esse papel |
| Dividir antes da padronização | A padronização (Pipeline, Etapa 4) aprende média e desvio só no treino |

---

## 6. Interpretação dos resultados obtidos

```
Formato do conjunto de treino: (142, 13)
Formato do conjunto de teste:  (36, 13)
Primeiros 15 rótulos do treino: [0 0 0 0 2 2 1 2 0 0 1 1 0 0 2]
```

142 + 36 = 178, então nenhuma amostra se perdeu. Os rótulos do treino estão misturados, o que mostra que o embaralhamento funcionou.

| Classe | Completo | Treino | Teste |
|---|---|---|---|
| class_0 | 59 (33,15%) | 47 (33,10%) | 12 (33,33%) |
| class_1 | 71 (39,89%) | 57 (40,14%) | 14 (38,89%) |
| class_2 | 48 (26,97%) | 38 (26,76%) | 10 (27,78%) |

- A estratificação funcionou: as proporções diferem menos de 1,1 ponto percentual entre os conjuntos. As pequenas diferenças vêm de as contagens serem números inteiros.
- O **teste tem 12, 14 e 10 amostras** de cada cultivar. Cada amostra vale ≈2,8 pontos de acurácia, e cada erro muda o recall da sua classe em 7 a 10 pontos.

---

## 7. Erros comuns e limitações

- **Dividir sem embaralhar** (por exemplo, pegar as últimas linhas como teste): o teste teria praticamente só `class_2`.
- **Esquecer a estratificação:** com 36 amostras no teste, a proporção de cada cultivar poderia variar bastante por acaso.
- **Padronizar antes de dividir:** seria vazamento.
- **Usar o teste para escolher o modelo:** o teste só será usado na Etapa 7.
- **Limitação:** um único teste de 36 amostras tem variância alta; outra divisão daria números diferentes.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que embaralhar?** O dataset vem ordenado por classe; sem embaralhar, o teste seria quase todo de uma cultivar.
2. **O que a estratificação garantiu?** As mesmas proporções das 3 cultivares no treino e no teste (diferenças menores que 1,1 p.p.).
3. **Por que 36 amostras no teste?** 20% de 178 = 35,6, arredondado para cima.
4. **Quanto vale cada amostra do teste?** ≈2,8 pontos de acurácia; e 7 a 10 pontos de recall da sua classe.
5. **Por que não um conjunto de validação separado?** O treino já é pequeno (142); a validação cruzada aproveita todo o treino.
6. **Como você sabe que embaralhou?** Os primeiros rótulos do treino estão misturados (`0 0 0 0 2 2 1 2 ...`).

---

## 9. Como apresentar oralmente

> "Separei 20% dos vinhos como teste, com o mesmo `train_test_split` estratificado da referência. Aqui dois cuidados foram essenciais: os dados vêm ordenados por classe, então a divisão precisa embaralhar, e o `train_test_split` faz isso por padrão, como mostram os primeiros rótulos do treino, já misturados. E como são só 178 amostras e três classes, estratifiquei para manter as proporções. Ficaram 142 vinhos para treino e 36 para teste: 12, 14 e 10 de cada cultivar. O teste é pequeno: cada vinho vale quase 3 pontos de acurácia, por isso a comparação principal entre os modelos vai ser feita com validação cruzada no treino."

Mostrar na tela: a célula do `train_test_split` com os rótulos embaralhados e a tabela de proporções.

---

## 10. Documentação oficial

Links para a versão *stable* de cada documentação (conferidos em 2026-10-12). O ambiente usa scikit-learn 1.9.1 e pandas 3.0.6. Se a versão *stable* do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**scikit-learn**

| Item | Para conferir | Link |
|---|---|---|
| `train_test_split` | `test_size`, `stratify`, `shuffle` (padrão `True`), `random_state` | [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| Estratificação | Divisões que preservam a proporção das classes | [scikit-learn — Cross-validation iterators with stratification](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) |
| `random_state` | Reprodutibilidade | [scikit-learn — Glossary: random_state](https://scikit-learn.org/stable/glossary.html#term-random_state) |
| Vazamento de dados | Dividir antes de transformar | [scikit-learn — Common pitfalls: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |

**pandas**

| Item | Para conferir | Link |
|---|---|---|
| `sort_index` | Classes em ordem na tabela | [pandas — DataFrame.sort_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html) |
| Divisão `DataFrame / Series` | Proporções por coluna | [pandas — DataFrame.div](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.div.html) |
| `DataFrame.map` | Formatação em porcentagem | [pandas — DataFrame.map](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.map.html) |
