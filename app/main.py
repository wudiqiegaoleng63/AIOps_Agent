from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from app.config import config
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, health, file, aiops
from fastapi.staticfiles import StaticFiles
import os
from app.core.milvus_client import milvus_manager
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    logger.info("=" * 60)
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")
    
    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")
    
    logger.info("=" * 60)
    
    yield
    
    # 关闭时执行
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")

    

app = FastAPI(
    title = config.app_name,
    description = "基于 LangChain 的智能OpenSre运维系统",
    version = config.app_version,
    lifespan=lifespan
)

#配置CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

#注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/chat", tags=["聊天"])
app.include_router(file.router, prefix="/file", tags=["文件管理"])
app.include_router(aiops.router, prefix="/aiops", tags=["AIOps运维"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )