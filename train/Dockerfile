FROM python:3.6.9

ARG HOME=/opt/mlflow

# Read build args from docker-compose.yml
ARG MLFLOW_HOST
ARG ARTIFACT_STORE
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID

# Assign build args to env vars so that they persist
ENV MLFLOW_HOST $MLFLOW_HOST
ENV ARTIFACT_STORE $ARTIFACT_STORE
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID

ARG MLFLOW_TRACKING_URI=http://${MLFLOW_HOST}:5000
ENV MLFLOW_TRACKING_URI $MLFLOW_TRACKING_URI
RUN echo "${MLFLOW_TRACKING_URI}"

RUN mkdir -p ${HOME}
COPY train/requirements.txt ${HOME}
ADD train ${HOME}/train

WORKDIR ${HOME}

EXPOSE 5000

RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=1000


RUN chmod +x ./train/run.sh
ENTRYPOINT ["./train/run.sh"]
