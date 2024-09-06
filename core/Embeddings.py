from typing import List
from ENV import API_KEY
class BaseEmbedding:
    def __init__(self, path:str, is_api:bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text:str, model: str) -> List[float]:
        raise NotImplementedError


class ZhipuEmbedding(BaseEmbedding):
    def __init__(self, path:str = "", is_api:bool = True, embedding_dim = 1024) -> None:
        super().__init__(path, is_api)
        if is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=API_KEY)
        self.embedding_dim = embedding_dim

    def get_embedding(self, text:str) -> List[float]:
        if self.client:
            response = self.client.embeddings.create(
                model = "embedding-2",
                input = text
            )
            return response.data[0].embedding
        else:
            raise NotImplementedError

