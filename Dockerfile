FROM python:3.8.0-slim 


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ls
ADD ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY app ./opt/app

WORKDIR /opt/app


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"



RUN chmod +x train &&\
    chmod +x test &&\
    chmod +x tune &&\
    chmod +x serve 
