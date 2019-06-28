#!/usr/bin/env bash

current_dir=$(pwd)
project_dir=$(dirname ${current_dir})

docker container run -it --rm -v ${project_dir}/notebooks/zeppelin/:/zeppelin/notebook -v ${project_dir}/datasets/:/zeppelin/datasets -p 8080:8080 -p 4040:4040 apache/zeppelin:0.8.1
