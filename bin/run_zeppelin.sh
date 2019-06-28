#!/usr/bin/env bash

docker container run -it --rm -v ../notebooks/zeppelin/:/zeppelin/notebook -v ../datasets/:/zeppelin/datasets -p 8080:8080 -p 4040:4040 apache/zeppelin:0.8.1
