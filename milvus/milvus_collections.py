from dotenv import load_dotenv
import os
from pymilvus import MilvusClient,DataType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from loguru import logger
import requests
import hashlib

load_dotenv()

milvus_uri = os.getenv("MILVUS_URI")
milvus_token = os.getenv("MILVUS_TOKEN")

milvus_client = MilvusClient(
    uri=milvus_uri,
    token=milvus_token
)

def create_collection(collection_name: str):
    if collection_name in milvus_client.list_collections():
        print(f"Collection '{collection_name}' already exists.")
        return
    schema = milvus_client.create_schema()

    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )

    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024
    )

    schema.add_field(
        field_name="context",
        datatype=DataType.VARCHAR,
        max_length=65535
    )

    schema.add_field(
        field_name="volume",
        datatype=DataType.VARCHAR,
        max_length=65535
    )

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"Collection '{collection_name}' created successfully.")
    

def read_context(file_path: str) -> list:
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    sentence_splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    context = [node.text for node in nodes]
    return context

if __name__ == "__main__":
    collection_name = "SilentWitchNovel"
    create_collection(collection_name)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(os.path.join(base_dir, "data", "SilentWitchNovel")):
        logger.info(f"Found file: {file}")
        file_path = os.path.join(
            base_dir,
            "data",
            "SilentWitchNovel",
            file
        )
        context = read_context(file_path)
        for i, c in enumerate(context):
            logger.info(f"Processing context {i+1}/{len(context)}")
            context_vector = requests.post(
                "http://127.0.0.1:26226/retrieval/v1/embedding/normal",
                json={"context": c},
                proxies={"http": None, "https": None}).json()["data"][0]["embedding"]
            milvus_client.insert(
                collection_name=collection_name,
                data=[
                    {
                        "id": hashlib.md5(c.encode("utf-8")).hexdigest(),
                        "vector": context_vector,
                        "context": c,
                        "volume": file
                    }
                ]
            )
            logger.info(f"Context:{c[:100]}...")
            logger.info(f"Inserted context {i+1}/{len(context)} into Milvus")
