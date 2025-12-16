from qdrant_client import QdrantClient,models

# Qdrant Config
SERVERQDRANT="http://222.255.214.30:6333"
COLLECTION_NAME="rag_document"
MODEL_EMBEDDING="keepitreal/vietnamese-sbert"


client= QdrantClient(url=SERVERQDRANT)


# Create collection in Qdrant
def creat_collection(client):

    vectors_config = models.VectorParams(size=768, distance=models.Distance.COSINE)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config
    )

# Insert embedding to Qdrant

# Search embedding in Qdrant

if __name__=="__main__":
    creat_collection(client)
    print('Embedding service')