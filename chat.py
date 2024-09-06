from idlelib.rpc import response_queue

from core import Embeddings, LLM, VectorBase

def main():
    vector_store = VectorBase.VectorBase()
    vector_store.load_vector("storage/github_data") #假设向量和文档已存储在该路径下

    embedding_model = Embeddings.ZhipuEmbedding()

    question = "Git中的文件有哪几种状态?"
    print(f"用户提问: {question}")

    # 在向量数据库中查找相关上下文
    context = vector_store.query(query=question, EmbeddingModel=embedding_model, k=1)[0]
    print(f"检索到的上下文：{context}")

    #初始化ZhipuAI聊天模型
    chat_model = LLM.GLMModel()

    response = chat_model.chat(prompt=question, history=[], content=context)
    print(f"模型回答：{response}")

if __name__ == "__main__":
    main()

