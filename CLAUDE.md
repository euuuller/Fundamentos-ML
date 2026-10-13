# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Coursework for a "Fundamentos de Machine Learning" course (IFMA). The repository holds **several exercises over time**; each one gets its own notebooks and its own guide folder. Deliverables are Jupyter notebooks plus study guides written in Portuguese (pt-BR): keep markdown, comments, plot titles, and printed messages in Portuguese.

### Repository layout (per exercise N)

| Path | Role |
|---|---|
| `Docs/reference/1-Arquivo.ipynb` | Instructor's reference notebook (MNIST "is it a 5?", binary), ending with the statements of Exercício 1 and 2. **Read-only.** Its style and section order are the model for solutions; it has known bugs that must not be copied (see `Docs/Guia/1-Exercise/00_Plano.md` § 1.5) |
| `Docs/reference/2-Arquivo.ipynb` | Instructor's reference for **multiclass** (MNIST 0–9: SGD/OvR, Dummy, RF, SVC/OvO; macro metrics), ending with the statement of Exercício 3 (Wine). **Read-only.** Its known bugs are listed in `Docs/Prompt/3-Prompt.md` (R — Rigor metodológico) |
| `Docs/Prompt/N-Prompt.md` | User's task specification for exercise N (`1-Prompt.md` = Exercise 01; binding restrictions: one stage per request, no test-set use before the final stage, no `assert` version checks, simple code). The user's file — don't edit its content; the paths inside `1-Prompt.md` predate the reorganization. `2-Prompt.md` = Exercise 02 (Titanic); `3-Prompt.md` = Exercise 03 (Wine) |
| `datasets/` | Data downloaded by notebooks (Titanic). Contents git-ignored; only `.gitkeep` is tracked. Notebooks reach it as `../datasets` |
| `Docs/Guia/N-Exercise/` | Guides for exercise N: `00_Plano.md` (plan, adaptations, per-stage workflow and guide template), `Status.md` (progress, next stage), `XX_Nome_da_Etapa.md` (one didactic guide per stage) |
| `Notebooks/` | **Study** notebooks: full solution with Markdown explanations (Exercise 01: `Notebooks/Exercicio_01_Cancer_Recall.ipynb`) |
| `Exercise/` | **Presentation** notebooks: the study notebook's code cells + section titles only (Exercise 01: `Exercise/1-Exercise.ipynb`) |
| `src/1_atividade/` | Only the `uv init` stub; not used |

**Before working on an exercise, read its `00_Plano.md` and `Status.md`, implement only the stage the user asked for, then update `Status.md` and stop.** New exercises follow the same pattern (`Docs/Prompt/2-Prompt.md`, `Docs/Guia/2-Exercise/`, a new study notebook in `Notebooks/`, `Exercise/2-Exercise.ipynb`); confirm names with the user.

### Study vs presentation notebooks

The presentation notebook is **generated**, never edited by hand, with `nbformat`:
- all code cells of the study notebook, in order, without outputs;
- from Markdown, **only titles**: the first line of a cell when it is a `#` heading or a fully bold line (`**...**`), excluding `**Observações:**`; drop titles that have no code cell before the next title of the same or higher level (bold = lowest level). Always keep the notebook title (first cell).

Then execute it with `nbconvert` and check that every code cell's text output matches the study notebook. Regenerate it whenever the study notebook changes. When comparing outputs between runs, concatenate consecutive stdout stream chunks first — Jupyter may split the same stdout into a varying number of chunks, which is not a real difference.

## Key facts for Exercício 01 (finished)

- `load_breast_cancer()` encodes **0 = malignant, 1 = benign**; the exercise requires Maligno = 1, so the target is `y = 1 - cancer.target`.
- SGD is wrapped in `make_pipeline(StandardScaler(), SGDClassifier(...))` so scaling is fit per CV fold (no leakage); Dummy and Random Forest use raw features.
- One shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for every `cross_validate`/`cross_val_predict`; decision thresholds were chosen by max F2 on out-of-fold training scores only; the test set was used once (section 5).
- Each guide ends with a "10. Documentação oficial" section of verified links (user request) — keep this in guides for future exercises too.

## Key facts for Exercício 02 — Titanic (finished; plan in `Docs/Guia/2-Exercise/00_Plano.md`)

- Files: spec `Docs/Prompt/2-Prompt.md`; study notebook `Notebooks/Exercicio_02_Titanic_Sobrevivencia.ipynb`; presentation `Exercise/2-Exercise.ipynb`; 10 stages.
- `load_titanic_data()` (reference) concatenates `train.csv` (891, labeled) and `test.csv` (418, **no `Survived`**) — keep only labeled rows. Use `Path("../datasets")`, since notebooks run from their own folder.
- Target `Survived` is already 1 = survived (549/342); still confirm it in the notebook.
- Features: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked` (drop `PassengerId, Name, Ticket, Cabin`). Missing values: `Age` 177, `Embarked` 2. Preprocessing goes in a `ColumnTransformer` inside the Pipeline (median imputation + `StandardScaler` for numerics incl. ordinal `Pclass`; most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')` for `Sex`, `Embarked`).
- Same 3 models and same `cv` as Exercise 01; metrics add **ROC/AUC**; decision threshold = **max F1** on out-of-fold training scores (no asymmetric error cost here).
- Describe feature–survival relations as associations, never causes. **No git commits** until the user creates the GitHub repo after Exercise 02.

## Key facts for Exercício 03 — Wine, multiclass (finished; spec `Docs/Prompt/3-Prompt.md`, plan `Docs/Guia/3-Exercise/00_Plano.md`)

- Files: plan/status/guides in `Docs/Guia/3-Exercise/`; study notebook `Notebooks/Exercicio_03_Wine_Multiclasse.ipynb`; presentation `Exercise/3-Exercise.ipynb`; 8 stages.
- `load_wine()`: 178 × 13, no missing values, target 0/1/2 = `class_0/1/2` (59/71/48), **rows ordered by class** — always shuffle. Built into scikit-learn (no download).
- 4 models from the reference (user decision): `SGDClassifier` (OvR) and `SVC(kernel='rbf', gamma='scale')` (OvO), each in a `Pipeline` with `StandardScaler`; `DummyClassifier(most_frequent)`; `RandomForestClassifier`. With 3 classes, OvR and OvO both train 3 binary classifiers.
- Metrics: accuracy + **macro** precision/recall/F1, per-class `classification_report`; same shared `StratifiedKFold(5, shuffle=True, random_state=42)`. **No ROC and no threshold tuning** (user decision); per-class analysis instead.
- Confusion matrices: on training, absolute + row-normalized + errors-only (`sample_weight=(pred != real)`), all **out-of-fold** via `cross_val_predict` (the reference wrongly uses in-sample `predict`); on test, row-normalized. TN/FP/FN/TP labels don't apply to multiclass.

## Environment & commands

uv-managed, Python 3.14 (`.python-version`), deps in `pyproject.toml` (numpy, pandas, scikit-learn, matplotlib, ipykernel, nbconvert). Windows. The environment is already set up — don't reinstall.

```sh
uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/Exercicio_01_Cancer_Recall.ipynb   # execute study notebook (clean kernel)
uv run jupyter nbconvert --to notebook --execute --inplace Exercise/1-Exercise.ipynb                    # execute presentation notebook
uv run jupyter nbconvert --to html Notebooks/Exercicio_01_Cancer_Recall.ipynb                          # export
uv add <pkg>                                                                                            # only if truly needed
```

There are no tests or linter.

When adding cells programmatically (nbformat), write the script to a file in the scratchpad and run it — don't pass it via a Bash heredoc/`-c` string: backslashes get unescaped on the way, which turned `"\n"` inside a cell into a real newline and LaTeX `\begin`/`\frac` into control characters. Use raw strings for LaTeX.
