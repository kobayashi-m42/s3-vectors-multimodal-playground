import base64
import glob
import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

model_id = "cohere.embed-v4:0"
vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME")
index_name = os.getenv("COHERE_INDEX_NAME")

def get_base64_image_uri(image_file_path: str, image_mime_type: str):
    with open(image_file_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{image_mime_type};base64,{base64_image}"

input_type = "search_document"
embedding_types = ["float"]
accept = '*/*'
content_type = 'application/json'
image_mime_type = "image/webp"

file_paths = glob.glob("../image/*")

embeddings = []
for path in file_paths:
    image_base64_uri = get_base64_image_uri(path, image_mime_type)

    body = json.dumps({
        "input_type": input_type,
        "images": [image_base64_uri],
        "embedding_types": embedding_types
    })

    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        accept=accept,
        contentType=content_type
    )
    response_body = json.loads(response['body'].read())
    embedding_vector = response_body["embeddings"]["float"][0]

    embeddings.append({
        "key": path,
        "data": {"float32": embedding_vector},
    })


res = s3vectors.put_vectors(
    vectorBucketName=vector_bucket_name, 
    indexName=index_name, 
    vectors=embeddings
)
