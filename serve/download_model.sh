#!/bin/bash

while getopts e:r:c: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;
        r) run=${OPTARG};;
        c) checkpoint=${OPTARG};;
    esac
done
echo "Username: $experiment";
echo "Age: $run";
echo "Full Name: $checkpoint";

echo "aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/lang_vocab.json ../"$LOCAL_MODEL_DIR"/lang_vocab.json"

aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/lang_vocab.json ../"$LOCAL_MODEL_DIR"/lang_vocab.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/char_vocab.json ../"$LOCAL_MODEL_DIR"/char_vocab.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/params.json ../"$LOCAL_MODEL_DIR"/params.json
aws s3 cp "$ARTIFACT_STORE"/"$experiment"/"$run"/artifacts/"$checkpoint" ../"$LOCAL_MODEL_DIR"/model --recursive
