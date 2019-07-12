#!/usr/bin/env bash

current_dir=$(pwd)
project_dir=$(dirname ${current_dir})

echo "WORK_DIR=${project_dir}" > ./.env

docker-compose -f ${project_dir}/env/docker/docker-compose.yml up
