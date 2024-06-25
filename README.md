# GRAPHADON

Repository for the 2024 Graphadon practical session on Graph Neural Networks

## Venv configuration

To make this practical course working correctly, please install the following venv using [poetry](https://python-poetry.org/) :

 * Copy the `pyproject.toml` file inside a working directory

 * in the working directory :

`$> export POETRY_VIRTUALENVS_IN_PROJECT=1; poetry install`

`$> poetry run python -m ipykernel install --name "graphadon" --user`

 * Now launch python and check if torch, torch_geometric and sklearn are available

 `$> poetry run python`

 `$> import torch`

 `$> import torch_geometric`

 `$> import networkx`

If no errors, everything is ok. If you have some errors :

 - check that you have a python >= 3.10 and <3.13

 - try to install torch and torch geometric by yourself : https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Now you should be able to launch a notebook : 

`$> poetry run jupyter notebook`

and choose the `graphadon` kernel.
