import base64
import json
import os

import boto3 
from dotenv import load_dotenv

load_dotenv()

model_id = "cohere.embed-v4:0"
bedrock= boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME")
index_name = os.getenv("COHERE_INDEX_NAME")

input_text = "寝ている猫"
input_type = "search_document"
embedding_types = ["float"]
accept = '*/*'
content_type = 'application/json'

body = json.dumps({
        "input_type": input_type,
        "texts": [input_text],
        "embedding_types": embedding_types
    })

response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        accept=accept,
        contentType=content_type
)

response_body = json.loads(response["body"].read())
embedding = response_body["embeddings"]["float"][0]


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
