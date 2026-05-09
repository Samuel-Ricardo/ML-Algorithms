# Laboratório 3: Algoritmos de Machine Learning

> **Status do projeto: Em desenvolvimento**
>
> Este repositório ainda está em construção. Algumas partes estão funcionais para estudo e experimentação, e outras ainda serão organizadas/refatoradas.

## Visão Geral

Projeto da disciplina **ES510 - Introdução à Inteligência Artificial (UFPE)** com implementações de:

- regressão linear com `scikit-learn`;
- regressão linear com gradiente descendente (implementação manual);
- regressão logística;
- árvore de decisão (dataset de tênis).

## Estrutura do Projeto

```text
03/
├── docs/
│   ├── [ES510]_Lab3.pdf
│   ├── dataset/
│   │   └── tennis.txt
│   └── ML/ML/02/
│       └── lr2_data.txt
├── src/
│   ├── model.py
│   ├── rl.py
│   ├── Decision Tree/
│   │   ├── model.py
│   │   └── temis.pdf
│   └── Regression/
│       ├── linear/
│       │   ├── model.py
│       │   └── rl.py
│       └── Logic/
│           └── model.py
└── requirements.txt
```

## Requisitos

Dependências em `requirements.txt`:

- scikit-learn
- matplotlib
- numpy
- pandas
- pydotplus
- ipykernel

## Como Configurar o Ambiente

Na raiz do projeto (`03/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como Executar

### 1) Regressão linear com scikit-learn

Arquivo: `src/model.py` (também em `src/Regression/linear/model.py`)

```powershell
python src/model.py
```

### 2) Regressão linear com gradiente descendente

Arquivo: `src/rl.py` (também em `src/Regression/linear/rl.py`)

```powershell
python src/rl.py
```

### 3) Regressão logística

Arquivo: `src/Regression/Logic/model.py`

```powershell
python "src/Regression/Logic/model.py"
```

### 4) Árvore de decisão

Arquivo: `src/Decision Tree/model.py`

```powershell
python "src/Decision Tree/model.py"
```

Observações:

- O script de árvore de decisão gera/atualiza o PDF `temis.pdf`.
- Caso o Graphviz (`dot`) não esteja instalado, o código usa `matplotlib` como fallback para gerar o PDF.

## Próximos Passos (Planejado)

- padronizar nomes de arquivos e pastas;
- melhorar organização entre código principal e versões de experimento;
- adicionar exemplos de entrada/saída esperada por script;
- incluir testes básicos;
- revisar caminhos relativos dos datasets para execução consistente.

## Aviso Importante

Como este projeto ainda está em desenvolvimento, a estrutura e os resultados podem mudar nas próximas versões.
