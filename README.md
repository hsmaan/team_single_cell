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
cd envs
conda env create -f env.yaml
```

The environment can be activated in an interactive session or a shell script:

```
conda activate single_cell_env
```

This environment contains all of the libraries that are necessary to get started on analyzing both the unimodal (RNA) and multi-modal single-cell sequencing data. Please start with the RNA data, going from experiment 5, then 7 and 8. 
