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

For statement 2, we will be using `en_core_web_sm` from spaCy.
```bash
python -m spacy download en_core_web_sm
```

Install notebook and ipywidgets with conda.
```bash
conda install notebook
```
```bash
conda install ipywidgets
```

Launch Jupyter notebook.
```bash
jupyter notebook
```


## 🔍 Analysis

The analysis are done in the following notebooks:
- Statement 1: [1-yields](./1-yields.ipynb)
- Statement 2: [2-feedback](./2-feedback.ipynb)


Processed data will be be stored in [`data/`](./data/) while results can be found in [`results/`](./results/)



## 💻 App

We use Streamlit as the tool for visualisation for Statement 2
```bash
streamlit run app.py
```
