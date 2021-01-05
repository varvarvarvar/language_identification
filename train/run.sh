#!/bin/bash

mlflow ui \
    --host $MLFLOW_HOST \
    --port 5000 \
    --default-artifact-root $ARTIFACT_STORE
