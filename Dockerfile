FROM python:3.10-slim
ENV PYTHONUNBUFFERED 1
ENV GIT_PYTHON_REFRESH quiet

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app/

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt && pip install spock-config==3.0.2 --no-deps

COPY . /app/

CMD [ "python3", "entry.py", "--config", "config/example_client/entry_client.yaml" ]