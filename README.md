# Project Description

The goal of this project was to fine tune various large language models on the task of research paper summarization. This repository contains code for training models and for testing them. 

# Installation Guide

For running pytorch models execute the following

### Pytorch Setup

```
    conda create --name torch python=3.9
    conda activate torch
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install datasets transformers matplotlib PyPDF2 
```

For running tensorflow models execute the following

### Tensorflow Setup

```
    conda create --name tf python=3.9
    conda activate tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install "tensorflow<2.11" 
    pip install datasets transformers matplotlib PyPDF2
```


### Environment Setup

Inside the working directory create a folder named data. Inside this folder create a cache folder and an experiments folder. Then for each model create a folder with the appropriate name(which can be found in the model file) in the experiments folder.

# Compilation Guide

To compile the models is relatively simple. Just run the following

```
conda activate environment_name
python3 model_name.py
```
