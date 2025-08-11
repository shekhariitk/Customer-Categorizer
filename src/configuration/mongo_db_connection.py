import os
import sys
import certifi
import pymongo
from dotenv import load_dotenv

from src.constant.database import DATABASE_NAME
from src.exception import CustomerException

# Load environment variables from .env file
load_dotenv()

ca = certifi.where()

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # Directly use the environment variable key string here
                mongo_db_url = os.getenv("MONGODB_URL")
                
                if mongo_db_url is None:
                    raise Exception("Environment variable 'MONGODB_URL' is not set.")
                
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise CustomerException(e, sys) from e
