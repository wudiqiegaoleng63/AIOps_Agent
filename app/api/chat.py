from fastapi import APIRouter
from app.models.request import ChatRequest
from loguru import logger
router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):

    try:
        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}")



    except Exception as e:
        