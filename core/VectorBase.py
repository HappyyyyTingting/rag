import json
import os.path
from time import sleep
from typing import List, AnyStr
from tqdm import tqdm
import numpy as np

class VectorBase:
    def __init__(self, document: List[str]=None) -> None:
        self.document = document if document else []
        self.vectors = []

    def get_vector(self, EmbeddingModel) -> List[List[float]]:
        """计算每个文档的向量"""
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating Embedding."):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """使用余弦相似度计算两个向量之间的相似度"""
        return np.dot(vector1, vector2) / (np.linalg.norm(vector2) * np.linalg.norm(vector1))

    def query(self, query:str, EmbeddingModel, k:int = 1) -> List[str]:
        """根据查询内容在向量数据库中找到最相似的文档"""
        query_vector = EmbeddingModel.get_embedding(query)
        similarity_scores = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])

        sorted_indices = np.argsort(similarity_scores)
        top_k_indices = sorted_indices[-k:][::-1]
        top_k_documents = np.array(self.document)[top_k_indices]
        return top_k_documents.tolist()

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/document.json", "w", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False)

        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f, ensure_ascii=False)


    def load_vector(self, path: str = "storage"):
        if os.path.exists(f"{path}/document.json"):
            with open(f"{path}/document.json", 'r', encoding="utf-8") as f:
                self.document = json.load(f)
        else:
            print(f"File {path}/document.json not found")

        if os.path.exists(f"{path}/vector.json"):
            with open(f"{path}/vector.json", 'r', encoding="utf-8") as f:
                self.vectors = json.load(f)
        else:
            print(f"File {path}/vector.json not found")



