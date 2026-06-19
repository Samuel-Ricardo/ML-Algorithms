# Projeto 03 - Algoritmos de Machine Learning

Projeto academico da disciplina ES510 (Introducao a IA - UFPE) com implementacoes de regressao linear, regressao logistica e arvore de decisao.

## Objetivo

Consolidar conceitos de ML classico em scripts independentes e comparaveis.

## Estrutura

```text
03/
  README.md
  requirements.txt
  docs/
    [ES510]_Lab3.pdf
    dataset/
      tennis.txt
    ML/ML/02/
      lr2_data.txt
  src/
    model.py
    rl.py
    Decision Tree/
      model.py
    Regression/
      linear/
        model.py
        rl.py
      Logic/
        model.py
```

## Requisitos

- Python 3.10+
- scikit-learn, matplotlib, numpy, pandas, pydotplus, ipykernel

## Setup

No PowerShell, na pasta 03:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como executar

Regressao linear:

```powershell
python src/model.py
```

Regressao linear com gradiente descendente:

```powershell
python src/rl.py
```

Regressao logistica:

```powershell
python "src/Regression/Logic/model.py"
```

Arvore de decisao:

```powershell
python "src/Decision Tree/model.py"
```

## Saidas esperadas

- metricas e visualizacoes por script;
- artefatos da arvore de decisao quando habilitados;
- comparacao de comportamentos entre abordagens.

## Observacoes

- execute scripts a partir da raiz 03 para manter caminhos relativos;
- se faltar Graphviz para exportacao de arvore, instale localmente;
- existem scripts duplicados em subpastas para estudo e organizacao por tema.

## Arquivos principais

- src/model.py — regressão linear
- src/rl.py — regressão linear com gradiente descendente
- src/knn.py — implementação do k-NN do zero (questão do Lab)
- src/Regression/Logic/model.py — regressão logística
- src/Decision Tree/model.py — árvore de decisão

## Questão do Lab

O arquivo src/knn.py resolve a questão 1 do PDF: implementação completa do classificador k-NN sem bibliotecas prontas, incluindo testes com diferentes valores de k e validação com dataset sintético.
