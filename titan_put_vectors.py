import base64
import glob
import json
import os

import boto3 
from dotenv import load_dotenv

load_dotenv()

bedrock= boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

modle_id = "amazon.titan-embed-image-v1"
vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME")
index_name = os.getenv("TITAN_NDEX_NAME")

file_paths = glob.glob("image/*")

embeddings=[]
for path in file_paths:
    image_bytes = open(path, "rb").read()

    body = json.dumps({
        "inputImage": base64.b64encode(image_bytes).decode("utf-8"),
        })    
    response = bedrock.invoke_model(
        modelId=modle_id,
        body=body
    )   
    response_body = json.loads(response['body'].read())
    embeddings.append({
            "key": path,
            "data": {"float32": response_body["embedding"]},
        })

res = s3vectors.put_vectors(
    vectorBucketName=vector_bucket_name, indexName=index_name, vectors=embeddings
)