#!/bin/bash

mlflow ui \
    --host $SERVER_HOST \
    --port $SERVER_PORT \
    --default-artifact-root $ARTIFACT_STORE