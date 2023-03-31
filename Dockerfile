FROM python:3.9.16-slim-bullseye

RUN apt-get update && apt-get upgrade -y

WORKDIR /app

COPY . .

RUN pip install -U pip setuptools &&\
    pip install -r requirements.txt

CMD ["bash"]
