#!/bin/bash

while getopts e:r:c: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;  # MLFlow experiment #
        r) run=${OPTARG};;  # MLFlow run ID 
        c) checkpoint=${OPTARG};;  # MLFlow
    esac
done

aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/lang_vocab.json ./"$LOCAL_MODEL_STORAGE"/lang_vocab.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/char_vocab.json ./"$LOCAL_MODEL_STORAGE"/char_vocab.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/params.json ./"$LOCAL_MODEL_STORAGE"/params.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/"$checkpoint" ./"$LOCAL_MODEL_STORAGE"/model --recursive
