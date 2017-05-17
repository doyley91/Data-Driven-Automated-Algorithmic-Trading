# -*- coding: utf-8 -*-

import pandas as pd
from sqlalchemy import create_engine

DB_TYPE = 'postgresql'
DB_DRIVER = 'psycopg2'
DB_USER = 'admin'
DB_PASS = 'password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'pandas_upsert'
POOL_SIZE = 50
TABLENAME = 'test_upsert'
SQLALCHEMY_DATABASE_URI = '%s+%s://%s:%s@%s:%s/%s' % (DB_TYPE, DB_DRIVER, DB_USER,
                                                      DB_PASS, DB_HOST, DB_PORT, DB_NAME)

ENGINE = create_engine(SQLALCHEMY_DATABASE_URI, pool_size=POOL_SIZE, max_overflow=0)

pd.to_sql(TABLENAME, ENGINE, if_exists='append', index=False)
