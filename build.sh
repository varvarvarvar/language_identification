#!/bin/bash

docker build -t mlflow .
docker run -d -p 5000:5000 --name mlflow mlflow
docker start mlflow