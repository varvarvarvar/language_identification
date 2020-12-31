#!/bin/bash

mlflow server \
    --backend-store-uri $FILE_STORE \
    --default-artifact-root $ARTIFACT_STORE \
    --host $SERVER_HOST \
    --port $SERVER_PORT