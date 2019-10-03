# Conda environment Setup

The conda environment is thinked to hold every piece of software needed to work in this project. 
Most of the tools are installed via the conda environment yaml file, but some of them requires additional setup, this guide will cover the environment setup.

## Table of contents
1. [Create Conda environment](#create-conda-environment)
2. [Complete installation of spark magic](#complete-installation-of-spark-magic)
3. [Copy the sparkmagic configuration file](#copy-the-sparkmagic-configuration-file)
4. [Install ipyleaflet lab extension](#install-ipyleaflet-lab-extension)

### Create conda environment
Open a terminal and type:
```bash
$ conda env create -n wtsp -f conda_env.yaml
```
### Complete installation of spark magic
After creating the environment, activate it and complete Spark magic installation:
```bash
$ conda activate wtsp
(wtsp) $ jupyter nbextension enable --py --sys-prefix widgetsnbextension 
(wtsp) $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
(wtsp) $ jupyter serverextension enable --py sparkmagic
```

### Copy the sparkmagic configuration file
```bash
$ mkdir -p ~/.sparkmagic/
$ cp ./config.json ~/.sparkmagic/
```

### Install ipyleaflet lab extension
```bash
$ conda activate wtsp
(wtsp) $ jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-leaflet
```