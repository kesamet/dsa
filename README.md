# dsa


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name dsa -f environment.yaml --force
```

Activate the environment.
```bash
conda activate dsa
```

Download `en_core_web_sm` from spaCy. It will be used in Problem 2.
```bash
python -m spacy download en_core_web_sm
```

Install notebook and ipywidgets with conda.
```bash
conda install notebook ipywidgets
```

Launch Jupyter notebook.
```bash
jupyter notebook
```


## 🔍 Analysis

The analysis are done in the following notebooks:
- Problem 1: [1-yields](./1-yields.ipynb)
- Problem 2: [2-feedback](./2-feedback.ipynb)


Data generated in the notebooks will be be stored in [`results/`](./results/).



## 💻 App

For Problem 2, we use Streamlit as the tool for visualisation.
```bash
streamlit run app.py
```
