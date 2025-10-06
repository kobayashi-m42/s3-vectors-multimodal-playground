import base64
import json
import os

import boto3 
from dotenv import load_dotenv

load_dotenv()

modle_id = "amazon.titan-embed-image-v1"
bedrock= boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME")
index_name = os.getenv("TITAN_NDEX_NAME")

input_text = "white cat"

body = json.dumps({
    "inputText": input_text,
    })

response = bedrock.invoke_model(
    modelId=modle_id,
    contentType="application/json",
    accept="application/json",
    body=body
)

response_body = json.loads(response["body"].read())
embedding = response_body["embedding"]

query = s3vectors.query_vectors(
    vectorBucketName=vector_bucket_name,
    indexName=index_name,
    queryVector={"float32":embedding},
    topK=3, 
    returnDistance=True,
    returnMetadata=True
)

results = query["vectors"]
print(results)
