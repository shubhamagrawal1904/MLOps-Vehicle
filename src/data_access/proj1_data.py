import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME, COLLECTION_NAME
from src.exception import MyException
from src.logger import logging

class Proj1Data:
    
    def __init__(self)->None:
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e,sys)
        
    def export_collection_as_dataframe(self, collection_name: str=COLLECTION_NAME, database_name :Optional[str]=None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]
            logging.info('Fetching Data from DB')
            logging.info(f'Printing \n*********\n Collection \n {collection}, {collection.count_documents({})}')
            df=pd.DataFrame(list(collection.find()))
            # print(df.head())
            
            logging.info(f'Data Fetched with Len : {len(df)}')
            
            if 'id' in df.columns:
                df=df.drop(columns=['id'])
                df.replace({'na':np.nan}, inplace=True)
                return df
        except Exception as e:
            raise MyException(e, sys)
        
        
            