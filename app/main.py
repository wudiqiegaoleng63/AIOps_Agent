from fastapi import FastAPI




@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    logger.info("=" * 60)

    

