from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]

def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    messages = state["messages"]

    if len(messages) <= 7:
        return None

    first_msg = messages[0]

    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]
    
    new_message = [first_msg] + list(recent_messages)

    logger.debug(f"修剪历史消息 {len(messages)} -> {len(new_message)}")

    return {
        "message" : [
            RemoveMessage(id = REMOVE_ALL_MESSAGES),
            *new_message

        ]
    }



class RagAgentService:
    def __init__(self, streaming: bool = True):
        self.model_name = config.dashscope_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()
        self.openai_flash_model = config.openai_flash_model
        self.openai_model = config.openai_model

        self.model = ChatOpenAI(
            model=self.openai_flash_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            streaming=streaming
        )

        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        self.mcp_tools: list = []

        self.checkpointer = MemorySaver()

        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成， model={self.openai_flash_model}, streaming={streaming}")


    async def _initialize_agent(self):
        if self._agent_initialized:
            return
        
        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(

            )
        

        pass




    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()
    
    async def query(
            self,
            question: str,
            session_id: str
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
          
        try: 
            await self._initializa_agent()

            logger.info(f"[会话 {session_id}]")
        
        except Exception as e:
                logger.error(
                    f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                    f"{format_exception_chain(e)}"
                )
                raise