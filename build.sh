#!/bin/bash

docker-compose --env-file .env build
docker run -d -p 5000:5000 --name mlflow language_identification_web
docker start mlflow