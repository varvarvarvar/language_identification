FROM python:3.6.9

# Read build args from docker-compose.yml
ARG MLFLOW_HOME
ARG ARTIFACT_STORE
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ARG LOCAL_MODEL_STORAGE

# Assign build args to env vars so that they persist
ENV MLFLOW_HOME $MLFLOW_HOME
ENV ARTIFACT_STORE $ARTIFACT_STORE
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV LOCAL_MODEL_STORAGE $LOCAL_MODEL_STORAGE

RUN mkdir -p ${MLFLOW_HOME}
ADD src ${MLFLOW_HOME}/src
ADD serve ${MLFLOW_HOME}/serve
COPY src/requirements.txt ${MLFLOW_HOME}

WORKDIR ${MLFLOW_HOME}

RUN pip3 install -r requirements.txt

RUN ["chmod", "+x", "serve/download_model.sh"]

EXPOSE 8080

ENTRYPOINT ["python3"]
CMD ["serve/main.py"]
