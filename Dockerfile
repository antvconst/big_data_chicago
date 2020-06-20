FROM jupyter/pyspark-notebook

RUN pip install sodapy
RUN pip install plotnine
RUN pip install plotly
RUN rmdir /home/jovyan/work

COPY hive-site.xml /usr/local/spark/conf

USER root

RUN mkdir /storage
RUN chmod 777 /storage
