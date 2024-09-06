from lib2to3.fixes.fix_input import context
from typing import List
from ENV import API_KEY
from zhipuai import ZhipuAI

ROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

class BaseModel:
    def __init__(self, path: str)->None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str ) -> str:
        pass

    def load_model(self, path):
        pass

class GLMModel(BaseModel):
    def __init__(self, path: str = "", model = "glm-4") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str ) -> str:
        client = ZhipuAI(api_key=API_KEY)
        history.append({
            "role":"user",
            "content":ROMPT_TEMPLATE["RAG_PROMPT_TEMPALTE"].format(question=prompt, context=content)
        })
        response = client.chat.completions.create(
            model = "glm-4",
            messages = history
        )
        return response.choices[0].message
