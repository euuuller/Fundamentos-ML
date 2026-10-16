# Etapa 1 — Importações e obtenção dos dados

Notebook: `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`, seções **1. Observar o Panorama Geral** e **2. Obtenção dos Dados** (início).
Correspondência na referência: seção EXERCÍCIO 2 de `Docs/reference/1-Arquivo.ipynb` (enunciado, dicionário de atributos e `load_titanic_data()`), além do `plt.rc` da seção 1.

---

## 1. Objetivo e relação com o exercício

Antes de qualquer análise, é preciso **definir o problema** e **ter os dados**. Esta etapa:

1. enquadra o problema: tipo de aprendizado, tarefa, entradas e saída, métricas (agora incluindo **curva ROC e AUC**);
2. registra o **dicionário de atributos** com os nomes reais das colunas;
3. baixa e carrega os dados com a função da referência, adaptada para salvar na pasta `datasets/` da raiz;
4. identifica e descarta os passageiros **sem rótulo** (do `test.csv`), que não servem para treinar nem avaliar.

---

## 2. Código implementado, por blocos

### Bloco 1 — Configuração dos gráficos

```python
import matplotlib.pyplot as plt
plt.rc('font', size=14)
...
```

Igual à referência e ao Exercício 01: tamanhos de fonte padrão para todos os gráficos.

### Bloco 2 — Panorama Geral e dicionário de atributos (Markdown)

Reproduz o enunciado do Exercício 2: supervisionado, classificação binária (0 = Não Sobreviveu, 1 = Sobreviveu), MISO, lote, métricas com ROC/AUC. Explica que, diferentemente do câncer, **não há um erro claramente mais grave**. O dicionário traz os **nomes exatos das colunas** do arquivo (`Pclass`, `SibSp`...), que diferem da grafia em minúsculas da referência.

### Bloco 3 — Função de carregamento (adaptada)

```python
PASTA_DATASETS = Path("../datasets")

def load_titanic_data():
    tarball_path = PASTA_DATASETS / "titanic.tgz"
    if not tarball_path.is_file():
        PASTA_DATASETS.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path=PASTA_DATASETS)

    train_df = pd.read_csv(PASTA_DATASETS / "titanic" / "train.csv")
    test_df = pd.read_csv(PASTA_DATASETS / "titanic" / "test.csv")
    return pd.concat([train_df, test_df], ignore_index=True)
```

A lógica é a da referência:
1. Se o `.tgz` ainda não existe, cria a pasta, baixa o arquivo e extrai.
2. Lê os dois CSVs.
3. Junta os dois num único `DataFrame`.

**Única adaptação:** o caminho virou a constante `PASTA_DATASETS = Path("../datasets")`. Caminhos relativos partem do diretório de execução, que é a pasta do notebook (`Notebooks/`). O `"datasets"` original criaria `Notebooks/datasets/`; com `../`, os dados vão para a pasta `datasets/` da raiz, criada e configurada no `.gitignore`.

### Bloco 4 — Carregamento e diagnóstico

```python
data = load_titanic_data()
print(sorted(p.name for p in (PASTA_DATASETS / "titanic").iterdir()))
print(f"Formato após o concat (train.csv + test.csv): {data.shape}")
print(f"Passageiros SEM valor em Survived: {data['Survived'].isnull().sum()}")
```

Mostra os arquivos extraídos, o formato após o `concat` e quantos passageiros não têm o alvo.

### Bloco 5 — Explicação (Markdown) e filtragem dos passageiros rotulados

```python
data = data.dropna(subset=['Survived']).reset_index(drop=True)
```

- `dropna(subset=['Survived'])` remove só as linhas com `Survived` vazio, sem olhar as outras colunas, que também têm ausentes legítimos (`Age`, `Cabin`).
- `reset_index(drop=True)` renumera as linhas de 0 a 890.

### Bloco 6 — Primeiras linhas e observações

`data.head()` mostra as 5 primeiras linhas. Uma célula Markdown resume os resultados.

---

## 3. Funções, parâmetros e variáveis relevantes

| Nome | Papel |
|---|---|
| `PASTA_DATASETS` | `Path("../datasets")`: pasta de dados na raiz do projeto |
| `Path.is_file()` | Verifica se o `.tgz` já foi baixado (evita baixar de novo) |
| `Path.mkdir(parents=True, exist_ok=True)` | Cria a pasta, se necessário, sem erro se já existir |
| `urllib.request.urlretrieve(url, destino)` | Baixa o arquivo da URL para o disco |
| `tarfile.open(...)` + `extractall(path=...)` | Abre o `.tgz` e extrai `titanic/train.csv` e `titanic/test.csv` |
| `pd.read_csv(...)` | Lê cada CSV como `DataFrame` |
| `pd.concat([...], ignore_index=True)` | Empilha os dois `DataFrame`s e renumera o índice |
| `data` | `DataFrame` final: **891 × 12**, só passageiros rotulados |
| `DataFrame.dropna(subset=[...])` | Remove linhas com ausente **apenas** nas colunas indicadas |
| `DataFrame.reset_index(drop=True)` | Renumera o índice e descarta o antigo |

---

## 4. Conceitos

- **Aprendizado supervisionado:** cada passageiro de treino tem a resposta (`Survived`), e o modelo aprende o mapeamento atributos → sobrevivência.
- **Classificação binária e classe positiva:** a classe 1 (Sobreviveu) é a positiva. **FN** é um sobrevivente previsto como não sobrevivente; **FP** é um não sobrevivente previsto como sobrevivente.
- **MISO:** vários atributos de entrada (numéricos e categóricos) e uma saída binária.
- **Dados rotulados × não rotulados:** só dados com rótulo permitem calcular erros. O `test.csv` do Kaggle existe para submissões à competição: o rótulo fica com o Kaggle. Sem ele, não há como medir acurácia, recall etc.
- **Caminho relativo:** é resolvido a partir do diretório de trabalho do processo. No Jupyter, por padrão, é a pasta do notebook. Por isso `../datasets` aponta para a raiz.
- **Tipo do alvo após o `concat`:** inteiros não podem representar "ausente" (`NaN`), então o pandas converteu `Survived` para `float` (0.0/1.0) ao juntar com linhas sem valor.

---

## 5. Justificativa das escolhas

| Escolha | Motivo |
|---|---|
| Manter a função da referência quase intacta | A especificação pede preservar o estilo da referência; a única mudança é o caminho |
| `PASTA_DATASETS` como constante | Deixa explícito onde os dados ficam e evita repetir o caminho em três lugares |
| Mostrar o problema do `concat` antes de filtrar | É mais didático evidenciar que existem 418 passageiros sem rótulo do que reescrever a função para ler só o `train.csv` |
| Filtrar com `dropna(subset=['Survived'])` | Remove exatamente as linhas sem alvo, sem descartar passageiros por ausentes em outras colunas |
| Dicionário com os nomes reais das colunas | Evita confusão entre `sibsp` (referência) e `SibSp` (arquivo) |
| Não converter `Survived` agora | A conferência da codificação do alvo é tarefa da Etapa 2 |
| Dados fora do git (`datasets/*` no `.gitignore`) | Dados baixados são reprodutíveis pela função e não precisam ser versionados |

---

## 6. Interpretação dos resultados obtidos

Saídas do notebook executado:

```
['test.csv', 'train.csv']
Formato após o concat (train.csv + test.csv): (1309, 12)
Passageiros SEM valor em Survived: 418
Formato após manter só os passageiros rotulados: (891, 12)
Passageiros SEM valor em Survived: 0
```

- O download e a extração funcionaram, e os arquivos estão em `datasets/titanic/`, na raiz do projeto. A pasta `Notebooks/datasets/` **não** foi criada, o que confirma a adaptação do caminho.
- O `concat` produziu **1309** passageiros, mas **418** deles (exatamente o tamanho do `test.csv`) não têm o alvo.
- Depois da filtragem, restam **891 passageiros e 12 colunas**, a base de todo o exercício, menor que os 569 × 30 do Exercício 01 em colunas e maior em linhas.
- O `head()` mostra a mistura de tipos que exigirá pré-processamento: números (`Age`, `Fare`), texto categórico (`Sex`, `Embarked`), texto livre (`Name`, `Ticket`) e ausentes já visíveis (`Cabin` = `NaN` em 3 das 5 primeiras linhas). `Survived` aparece como `0.0`/`1.0`.
- O `git status` confirmou que o conteúdo baixado em `datasets/` é ignorado e que apenas `datasets/.gitkeep` aparece.

---

## 7. Erros comuns e limitações

- **Usar o `concat` inteiro para treinar:** as 418 linhas sem alvo quebrariam o treino ou, pior, seriam tratadas de forma errada.
- **Remover linhas com `dropna()` sem `subset`:** descartaria quase todos os passageiros, porque `Cabin` está ausente em 77% deles.
- **Caminho relativo errado:** com `"datasets"`, os dados iriam para `Notebooks/datasets/`, fora do `.gitignore` e duplicados por pasta.
- **Esquecer que `Survived` virou `float`:** comparações como `y == 1` funcionam, mas o tipo correto (inteiro) será restabelecido na Etapa 2.
- **Dependência de internet:** a primeira execução precisa baixar o arquivo. Depois, a função usa a cópia local (`is_file()`).
- **Limitação:** 891 passageiros é um conjunto pequeno. As estimativas de desempenho terão incerteza considerável, como no Exercício 01.

---

## 8. Perguntas que o professor poderia fazer

1. **Por que o conjunto passou de 1309 para 891 linhas?** Porque os 418 passageiros do `test.csv` não têm `Survived`; sem rótulo, não servem para treinar nem avaliar.
2. **Então como você vai testar o modelo?** Separando um conjunto de teste estratificado a partir dos 891 rotulados (seção 2.3).
3. **Por que `../datasets` e não `datasets`?** O notebook roda na pasta `Notebooks/`; `../` sobe para a raiz, onde fica a pasta de dados ignorada pelo git.
4. **Por que `Survived` aparece como 0.0 e 1.0?** O `concat` introduziu valores ausentes, e inteiros não representam `NaN` no pandas, então a coluna virou `float`.
5. **Qual a classe positiva?** Sobreviveu (1). FN é um sobrevivente que o modelo diz que não sobreviveu.
6. **Por que não usar `dropna()` sem argumentos?** Removeria passageiros com ausentes em `Age` ou `Cabin`, perdendo a maior parte dos dados.

---

## 9. Como apresentar oralmente

> "O Exercício 2 é uma classificação binária supervisionada: prever se um passageiro do Titanic sobreviveu. As métricas incluem, além das do primeiro exercício, a curva ROC e a AUC. Usei a função de carregamento da referência, com uma adaptação no caminho, para os dados ficarem na pasta `datasets` da raiz, que é ignorada pelo git. A função junta o `train.csv` e o `test.csv`, mas o `test.csv` é o arquivo de submissão do Kaggle e não tem a coluna de sobrevivência. Das 1309 linhas, 418 não têm rótulo. Por isso mantive só os 891 passageiros rotulados; o teste será separado deles na próxima seção. As primeiras linhas já mostram a mistura de números, texto e valores ausentes que vamos tratar."

Mostrar na tela: o Panorama Geral, a saída `(1309, 12)` → `418` → `(891, 12)` e o `head()`.

---

## 10. Documentação oficial

Links para a versão *stable* ou atual de cada documentação (conferidos em 2026-10-12). O ambiente usa Python 3.14, pandas 3.0.6 e matplotlib 3.11.2. Se a versão do site for mais nova, o seletor de versão no topo de cada página leva à documentação da versão instalada.

**Dados**

| Item | Para conferir | Link |
|---|---|---|
| Titanic no Kaggle | Descrição oficial dos arquivos `train.csv` e `test.csv` (este sem `Survived`) e do dicionário de atributos | [Kaggle — Titanic: Data](https://www.kaggle.com/competitions/titanic/data) |
| Repositório de origem do `.tgz` | Repositório do livro *Hands-On Machine Learning* (3ª ed.), de onde vem a função de carregamento | [GitHub — ageron/handson-ml3](https://github.com/ageron/handson-ml3) |

**Python (biblioteca padrão)**

| Item | Para conferir | Link |
|---|---|---|
| `pathlib` | Caminhos com `Path` e o operador `/` | [Python — pathlib](https://docs.python.org/3/library/pathlib.html) |
| `Path.is_file` | Verificar se o arquivo já existe | [Python — Path.is_file](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file) |
| `Path.mkdir` | `parents` e `exist_ok` | [Python — Path.mkdir](https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir) |
| `Path.iterdir` | Listar os arquivos extraídos | [Python — Path.iterdir](https://docs.python.org/3/library/pathlib.html#pathlib.Path.iterdir) |
| `urlretrieve` | Baixar o arquivo | [Python — urllib.request.urlretrieve](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlretrieve) |
| `tarfile` | Abrir arquivos `.tgz` | [Python — tarfile](https://docs.python.org/3/library/tarfile.html) |
| `TarFile.extractall` | Extração para uma pasta | [Python — TarFile.extractall](https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall) |
| Filtros de extração | Comportamento de segurança na extração | [Python — Extraction filters](https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter) |

**pandas e Matplotlib**

| Item | Para conferir | Link |
|---|---|---|
| `read_csv` | Leitura dos arquivos | [pandas — read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| `concat` | Empilhar `DataFrame`s e `ignore_index` | [pandas — concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) |
| `isnull` | Contar ausentes | [pandas — DataFrame.isnull](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) |
| `dropna` | Parâmetro `subset` | [pandas — DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html) |
| `reset_index` | Parâmetro `drop` | [pandas — DataFrame.reset_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html) |
| `head` | Primeiras linhas | [pandas — DataFrame.head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) |
| `plt.rc` | Configuração dos gráficos | [Matplotlib — pyplot.rc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html) |

**Conceitos e projeto**

| Item | Para conferir | Link |
|---|---|---|
| Aprendizado supervisionado | Visão geral no scikit-learn | [scikit-learn — Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) |
| `.gitignore` | Padrões `datasets/*` e a exceção `!datasets/.gitkeep` | [Git — gitignore](https://git-scm.com/docs/gitignore) |
