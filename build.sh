#!/bin/bash

docker-compose --env-file .env build

docker run -d \
    -p 5000:5000 \
    --mount type=bind,source="$(pwd)"/mlruns,target=/opt/mlflow/mlruns \
    --mount type=bind,source="$(pwd)"/input,target=/opt/mlflow/input \
    --name mlflow \
    language_identification_web

docker start mlflow

docker run -d \
    -p 8080:8080 \
    --name serve \
    language_identification_serve

docker start serve
