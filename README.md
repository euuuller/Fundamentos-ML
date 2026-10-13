# Fundamentos de Machine Learning — Atividades (IFMA)

Repositório das atividades práticas da disciplina. Cada exercício tem uma **especificação** (`Docs/Prompt/N-Prompt.md`), um notebook de **estudo** (solução completa e comentada), um notebook de **apresentação** (código e títulos) e uma pasta de **guias** didáticos.

## Estrutura

```
Docs/
  reference/
    1-Arquivo.ipynb             referência do professor: classificação binária (MNIST) + enunciados dos Exercícios 1 e 2
    2-Arquivo.ipynb             referência do professor: classificação multiclasse (MNIST) + enunciado do Exercício 3
  Prompt/
    1-Prompt.md                 especificação do Exercício 01
    2-Prompt.md                 especificação do Exercício 02
    3-Prompt.md                 especificação do Exercício 03
  Guia/
    1-Exercise/                 guias do Exercício 01 (plano, status e um guia por etapa)
    2-Exercise/                 guias do Exercício 02
    3-Exercise/                 guias do Exercício 03
Notebooks/
  Exercicio_01_Cancer_Recall.ipynb         Exercício 01: notebook de estudo
  Exercicio_02_Titanic_Sobrevivencia.ipynb Exercício 02: notebook de estudo
  Exercicio_03_Wine_Multiclasse.ipynb      Exercício 03: notebook de estudo
Exercise/
  1-Exercise.ipynb                   Exercício 01: notebook de apresentação (código + títulos)
  2-Exercise.ipynb                   Exercício 02: notebook de apresentação (código + títulos)
  3-Exercise.ipynb                   Exercício 03: notebook de apresentação (código + títulos)
datasets/                       dados baixados pelos notebooks (conteúdo ignorado pelo git;
                                o Exercício 02 baixa o Titanic na primeira execução)
```

Novas atividades seguem o mesmo padrão: `Docs/Prompt/N-Prompt.md`, `Docs/Guia/N-Exercise/`, um notebook de estudo em `Notebooks/` e um de apresentação em `Exercise/N-Exercise.ipynb`.

## Exercícios

| Nº | Tema | Estudo | Apresentação | Guias |
|---|---|---|---|---|
| 01 | Diagnóstico de Câncer e Otimização de Recall (`load_breast_cancer`; SGD, Dummy, Random Forest) | [Notebook](Notebooks/Exercicio_01_Cancer_Recall.ipynb) | [Notebook](Exercise/1-Exercise.ipynb) | [Docs/Guia/1-Exercise](Docs/Guia/1-Exercise/Status.md) |
| 02 | Previsão de Sobrevivência no Titanic (SGD, Dummy, Random Forest; ROC/AUC) | [Notebook](Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb) | [Notebook](Exercise/2-Exercise.ipynb) | [Docs/Guia/2-Exercise](Docs/Guia/2-Exercise/Status.md) |
| 03 | Classificação Multiclasse de Vinhos (`load_wine`; SGD/OvR, Dummy, Random Forest, SVC/OvO; métricas macro) | [Notebook](Notebooks/Exercicio_03_Wine_Multiclasse.ipynb) | [Notebook](Exercise/3-Exercise.ipynb) | [Docs/Guia/3-Exercise](Docs/Guia/3-Exercise/Status.md) |

## Ambiente

Projeto gerenciado com [uv](https://docs.astral.sh/uv/) (Python 3.14).

```sh
uv sync                                                                                   # cria/atualiza o ambiente
uv run jupyter nbconvert --to notebook --execute --inplace Exercise/1-Exercise.ipynb      # executa um notebook do início ao fim
```

Os resultados são acadêmicos: no Exercício 01, sem validade para uso clínico; no Exercício 02, as relações encontradas são associações nos dados, não causas.
