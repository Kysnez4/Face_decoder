import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingDatabase:
    def __init__(self, json_file='embeddings.json'):
        self.json_file = json_file
        self.embeddings = defaultdict(list)
        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                for name, embeddings in data.items():
                    self.embeddings[name] = [np.array(embedding) for embedding in embeddings]

    def save_embeddings(self):
        with open(self.json_file, 'w') as f:
            data = {name: [embedding.tolist() for embedding in embeddings] for name, embeddings in self.embeddings.items()}
            json.dump(data, f, indent=4)

    def insert_embedding(self, name, embedding):
        self.embeddings[name].append(embedding)
        self.save_embeddings()

    def delete_embedding(self, name):
        if name in self.embeddings:
            del self.embeddings[name]
            self.save_embeddings()

    def find_nearest_embeddings(self, query_embedding, top_k=5, similarity_threshold=0.5):
        query_embedding = query_embedding.reshape(1, -1)
        results = []
        for name, embeddings in self.embeddings.items():
            for embedding in embeddings:
                embedding = embedding.reshape(1, -1)
                similarity = cosine_similarity(query_embedding, embedding)[0][0]
                if similarity >= similarity_threshold:
                    results.append((name, name, similarity))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]