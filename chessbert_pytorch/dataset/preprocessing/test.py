import pinecone
import numpy as np
import time

pinecone.init(api_key = '38132697-8f87-4930-a355-376bd93394a3', environment = "us-east4-gcp")
index = pinecone.Index('chesspos-lichess-embeddings')

queries = np.random.randn(40, 64)

s = time.time()
result = index.query(queries=queries.tolist(), top_k = 8, include_metadata = True)
print(type(result))
print(type(result['results']))
print(type(result['results'][0]))



