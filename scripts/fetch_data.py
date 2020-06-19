import functools
import multiprocessing
import logging
import gc

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import pyspark
import pyspark.sql
from pyspark.sql.types import StructType, StringType
from sodapy import Socrata


DOMAIN = 'data.cityofchicago.org'
APP_TOKEN = 'nHT9Ne9JPaDCYDkaNunZHlKtI'
DATASET = 'ijzp-q8t2'  # Chicago Crime dataset id 

HIVE_WAREHOUSE_DIR = '/storage'
HIVE_DATABASE = 'chicago_crime'
HIVE_TABLE = f"{HIVE_DATABASE}.records"

NUM_WORKERS = 4
BATCH_SIZE = 1000

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_schema():
    return StructType(). \
        add('id', StringType(), False). \
        add('case_number', StringType(), False). \
        add('date', StringType(), False). \
        add('block', StringType()). \
        add('iucr', StringType()). \
        add('primary_type', StringType()). \
        add('description', StringType()). \
        add('location_description', StringType()). \
        add('arrest', StringType()). \
        add('domestic', StringType()). \
        add('beat', StringType()). \
        add('district', StringType()). \
        add('ward', StringType()). \
        add('community_area', StringType()). \
        add('fbi_code', StringType()). \
        add('latitude', StringType()). \
        add('longitude', StringType())

def enforce_types(df):
    return df.withColumn('date', df['date'].cast('date')). \
              withColumn('arrest', df['arrest'].cast('boolean')). \
              withColumn('domestic', df['domestic'].cast('boolean')). \
              withColumn('beat', df['beat'].cast('integer')). \
              withColumn('district', df['district'].cast('integer')). \
              withColumn('ward', df['ward'].cast('integer')). \
              withColumn('community_area', df['community_area'].cast('integer')). \
              withColumn('latitude', df['latitude'].cast('float')). \
              withColumn('longitude', df['longitude'].cast('float'))

def query_size(soda_client, dataset):
    # q = soda_client.get(dataset, query="select count(*) where domestic=true")
    q = soda_client.get(dataset, query="select count(*)")
    return int(q[0]['count'])

def fetch_batch(batch_idx, soda_client, batch_size):
    offset = batch_idx * batch_size
    batch = soda_client.get(
        DATASET, limit=batch_size, content_type='csv',
        select='id, case_number, date, block, iucr,\
                primary_type, description, location_description,\
                arrest, domestic, beat, district, ward, community_area,\
                fbi_code, latitude, longitude'#,
        # domestic=True  # we will extract only domestic crimes
    )
    return batch_idx, batch[1:]

def main():
    logger.info('Creating Spark session')
    spark = pyspark.sql.SparkSession.builder. \
        master('local[*]'). \
        config("spark.sql.warehouse.dir", HIVE_WAREHOUSE_DIR). \
        config("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", True). \
        enableHiveSupport(). \
        getOrCreate()  # create spark session to dump the data into Hive
    logger.info('Initializing Hive database')
    spark.sql(f'drop database if exists {HIVE_DATABASE} cascade')
    spark.sql(f'create database {HIVE_DATABASE}')

    logger.info('Accessing remote dataset')
    soda_client = Socrata(DOMAIN, APP_TOKEN)  # create client for accessing API
    soda_client.timeout = 50  # otherwise we will get a lot of timeout errors
    dataset_size = query_size(soda_client, DATASET)
    num_batches = dataset_size // BATCH_SIZE + 1
    logger.info('Remote dataset size: %d (%d batches)', dataset_size, num_batches)

    schema = make_schema()  # spark dataframe schema for our data

    logger.info('Starting process pool')
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        fetch_batch_partial = functools.partial(
            fetch_batch, soda_client=soda_client, batch_size=BATCH_SIZE
        )
        batch_it = pool.imap_unordered(
            fetch_batch_partial, range(num_batches)
        )
        logger.info('Fetching data')
        for idx, batch in tqdm(batch_it, total=num_batches):
            df = spark.createDataFrame(batch, schema=schema)
            df = enforce_types(df)
            df.createOrReplaceTempView('tmp_table')
            if spark.catalog._jcatalog.tableExists(HIVE_TABLE):
                spark.sql(f'insert into {HIVE_TABLE} from tmp_table')
            else:
                spark.sql(f'create table {HIVE_TABLE} as select * from tmp_table')

    logger.info('Probing the database')
    df = spark.sql(f"select * from {HIVE_TABLE} limit 200")
    logger.info(f'Received {df.count()} rows from {HIVE_TABLE}')

    soda_client.close()
    spark.stop()
    logger.info('Fetching finished')
    logger.info(f'All data has been written to Hive table {HIVE_TABLE}')



if __name__ == "__main__":
    main()
