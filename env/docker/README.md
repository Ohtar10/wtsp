# Docker Environment setup

This folder contains the docker artifacts that wraps the cli to train the models.

## Contents
### [Dockerfile](Dockerfile)
The docker image file definition to create containers that has already the tool installed.

### [docker-compose](docker-compose.yml)
The docker compose file with some system property definitions for the runtime. You will notice that there is an external property called `WORKDIR` that is not defined, this is the path to your host file system where you have the datasets or where you want to have the outputs for the process. To define this property you can either export this variable as an environment property via `export WORKDIR=your_path`, or create a `.env` file along with the docker compose file where you can define this variable.

## Usage
Take a look at [all-cli-process](../../notebooks/jupyter/machine-learning/all-cli-process.ipynb) note for full usage examples.