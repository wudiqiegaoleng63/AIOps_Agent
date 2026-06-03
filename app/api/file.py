from fastapi import APIRouter
from app.config import config
from typing import Any
from app.core.milvus_client import milvus_manager
from fastapi.responses import JSONResponse
from loguru import logger

router = APIRouter()






