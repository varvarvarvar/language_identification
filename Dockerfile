FROM python:3.8

# Read build args from docker-compose.yml
ARG MLFLOW_HOME
ARG SERVER_PORT
ARG SERVER_HOST
ARG ARTIFACT_STORE
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID

# Assign build args to env vars so that they persist
ENV MLFLOW_HOME $MLFLOW_HOME
ENV SERVER_PORT $SERVER_PORT
ENV SERVER_HOST $SERVER_HOST
ENV ARTIFACT_STORE $ARTIFACT_STORE
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID

RUN mkdir -p ${MLFLOW_HOME}
COPY src/requirements.txt ${MLFLOW_HOME}
ADD src ${MLFLOW_HOME}/src
ADD input ${MLFLOW_HOME}/input

EXPOSE ${SERVER_PORT}/tcp

WORKDIR ${MLFLOW_HOME}

RUN pip3 install -r requirements.txt

RUN ["chmod", "+x", "./src/run.sh"]
ENTRYPOINT ["./src/run.sh"]