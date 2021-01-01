FROM python:3.8

ENV MLFLOW_HOME /opt/mlflow
ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0
# ENV FILE_STORE ${MLFLOW_HOME}/src/fileStore
# ENV ARTIFACT_STORE ${MLFLOW_HOME}/src/artifactStore

RUN mkdir -p ${MLFLOW_HOME}/src
# RUN mkdir -p ${MLFLOW_HOME}/src && \
#     mkdir -p ${FILE_STORE} && \
#     mkdir -p ${ARTIFACT_STORE}

COPY src/requirements.txt ${MLFLOW_HOME}
ADD src ${MLFLOW_HOME}/src
ADD input ${MLFLOW_HOME}/input
# COPY src/run.sh ${MLFLOW_HOME}/src/run.sh

EXPOSE ${SERVER_PORT}/tcp

# VOLUME ["${MLFLOW_HOME}/src/", "${FILE_STORE}", "${ARTIFACT_STORE}"]

WORKDIR ${MLFLOW_HOME}

# COPY .env ${MLFLOW_HOME}

RUN pip3 install -r requirements.txt

RUN ["chmod", "+x", "./src/run.sh"]
ENTRYPOINT ["./src/run.sh"]
