#!/usr/bin/env bash

current_dir=$(pwd)
project_dir=$(dirname ${current_dir})

docker-compose -f ${project_dir}/env/docker/docker-compose.yml down

rm ./.env