from langchain_openai import ChatOpenAI
from app.config import config
from loguru import logger



class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    #  OpenAI pro兼容模式 URL

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        model = model or config.openai_model
        base_url = base_url or config.openai_base_url
        api_key = api_key or config.openai_api_key

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
        )

        return llm

# 全局 LLM 工厂实例
llm_factory = LLMFactory()