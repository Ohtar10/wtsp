# Conda environment Setup

The conda environment contains the main software packages to work with the project. Most of the tools are installed via the conda environment yaml file, but some of them requires additional setup, this guide will cover the environment setup.

## Notes:
* During the development of this project, creating a conda environment using a recipie with all the dependencies proved to be difficult since conda is not always able to find a suitable compatibility matrix for the dependencies, so sometimes it is advisable to create a vanilla conda environment with python 3.7, and then installing the dependencies by hand.
* The `wtsp-linux-full.yaml` conda environment file is mostly for reference since, again, it is very difficult that conda is able to find a compatibility matrix for all the listed dependencies.

## Table of contents
1. [Create Conda environment](#create-conda-environment)
2. [Complete installation of spark magic](#complete-installation-of-spark-magic)
3. [Copy the sparkmagic configuration file](#copy-the-sparkmagic-configuration-file)

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
$ cp ./sparkmagic.config.json ~/.sparkmagic/
```