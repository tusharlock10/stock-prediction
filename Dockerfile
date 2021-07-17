FROM tensorflow/tensorflow

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD ["server.py"]