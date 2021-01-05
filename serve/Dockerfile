FROM python:3.6.9

ARG HOME=/opt/flask

# Read build args from docker-compose.yml
ARG ARTIFACT_STORE
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ARG LOCAL_MODEL_STORAGE

# Assign build args to env vars so that they persist
ENV HOME $HOME
ENV ARTIFACT_STORE $ARTIFACT_STORE
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV LOCAL_MODEL_STORAGE $LOCAL_MODEL_STORAGE

RUN mkdir -p ${HOME}
ADD train ${HOME}/train
ADD serve ${HOME}/serve
COPY serve/requirements.txt ${HOME}

WORKDIR ${HOME}

EXPOSE 8080

RUN pip3 install -U -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=1000

RUN chmod +x serve/download_model.sh

ENTRYPOINT ["python3"]
CMD ["./serve/main.py"]
