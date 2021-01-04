#!/bin/bash

aws s3 cp s3://aws-language-identification/artifacts/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/lang_vocab.json ../artifacts/lang_vocab.json
aws s3 cp s3://aws-language-identification/artifacts/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/char_vocab.json ../artifacts/char_vocab.json
aws s3 cp s3://aws-language-identification/artifacts/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/params.json ../artifacts/params.json
aws s3 cp s3://aws-language-identification/artifacts/0/e9b1c8c553904e19b1a7b3564a1ee8cf/artifacts/49.model ../artifacts/model --recursive
