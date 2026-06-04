from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from app.services.vector_index_service import vector_index_service
from loguru import logger
router = APIRouter()


# 文件上传后存储的路径
UPLOAD_DIR = Path("./uploads")
# 支持的文件类型
ALLOWED_EXTENSIONS = ["txt", "md"]
# 单个文件支持最大大小
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        safe_filename = _sanitize_filename(file.filename)

        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的格式，支持：{', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        #创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file_path = UPLOAD_DIR / safe_filename

        if file_path.exists():
            logger.info(f"文件已经存在, 将覆盖: {file_path}")
            file_path.unlink()

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件大小超过限制 最大{MAX_FILE_SIZE} 字节")
        
        file_path.write_bytes(content)

        logger.info(f"文件上传成功: {file_path}")

        #创建向量索引

        try:
            vector_index_service.index_single_file(str(file_path))
        except Exception as e:
            logger.error(f"向量索引创建失败 {e}")

        # 6. 返回响应
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


@router.post("/index_directory")
async def index_directory(directory_path: Optional[str] = None):
    """
    索引指定目录下的所有文件

    Args:
        directory_path: 目录路径（可选，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果
    """
    try:
        logger.info(f"开始索引目录: {directory_path or 'uploads'}")

        # 执行索引
        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")




def _sanitize_filename(filename: str) -> str:
    santized = filename.replace(" ", "_")

    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        santized = santized.replace(char, "_")
    return santized

def _get_file_extension(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    if len(parts) >= 2:
        return parts[1].lower()
    return ""