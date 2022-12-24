## Team single-cell - integration of multi-modal single-cell sequencing data

***

### Repository for the joint UofT/UCU collaborative project course - Fall 2022

***

### Instructions:

#### Data:

The data will be available on the Vector cluster, but if you'd like access sooner, please just ping the Slack and I can add the download instructions here. It's not recommended to analyze the data on a personal computer.

#### Environment:

To set up the environment for analyzing single-cell data and method development, please download miniconda (https://docs.conda.io/en/latest/miniconda.html) for your system. Once Vector access is obtained, miniconda can be downloaded on the Vector linux system.

Once miniconda is set up, install the environment in the following manner:

```
conda install mamba -n base -c conda-forge 
cd envs
mamba env create -f env.yaml
```

We'll use mamba to install and manage environments as it's much a better solver.

The environment can be activated in an interactive session or a shell script:

```
conda activate single_cell_env
```

This environment contains all of the libraries that are necessary to get started on analyzing both the unimodal (RNA) and multi-modal single-cell sequencing data. Please start with the RNA data, going from experiment 5, then 7 and 8. 

#### Jupyter notebooks 

Jupyter is installed in the single_cell_env environment. To use this conda environment with jupyter, add the kernelspec to your jupyter path with the following:

```
conda activate single_cell_env 
python -m ipykernel install --user --name "single_cell_env" --display-name "single_cell_env"
```

We used specific versions of torch and cuda to train our models which can be installed by performing the following command:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
```
Note that this version of torch and cuda could uncompatible with your system settings.

Now you should be able to launch jupyter notebooks and use the single_cell_env kernel:

```
jupyter-notebook
```
