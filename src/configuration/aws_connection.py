import os
import boto3
from src.constant.env_variable import  REGION_NAME
from dotenv import load_dotenv

load_dotenv()  # load from .env if present

class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            access_key_id = os.getenv("AWS_ACCESS_KEY_ID")  # use actual string keys here directly
            secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if access_key_id is None:
                raise Exception("Environment variable 'AWS_ACCESS_KEY_ID' is not set.")
            if secret_access_key is None:
                raise Exception("Environment variable 'AWS_SECRET_ACCESS_KEY' is not set.")

            S3Client.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )
            S3Client.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
