FROM python:3.9

USER root

ADD requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY ./src ./src
COPY ./models ./models
COPY ./tests ./tests 
