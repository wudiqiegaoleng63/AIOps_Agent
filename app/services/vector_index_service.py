"""向量索引服务"""
from datetime import datetime
from typing import Dict, Optional, Any
from loguru import logger
from pathlib import Path
from app.services.vector_store_manager import vector_store_manager
from app.services.document_splitter_service import document_splitter_service
class IndexingResult:
    """索引结果"""
    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}
    
    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files
        }

class VectorIndexService:

    def __init__(self):
        self.upload_path = "./uploads"
        logger.info("📁 VectorIndexService 初始化完成")
    
    def index_directory(self, directory_path: Optional[str]) -> IndexingResult:
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result
    def index_single_file(self, file_path: str):
        """索引单个文件的内容到 Milvus"""
        # 这里应该包含实际的文件读取和向量化逻辑
        # 例如：
        # 1. 读取文件内容
        # 2. 使用文本分割器将内容分块
        # 3. 将每个块转换为向量
        # 4. 将向量存储到 Milvus 中
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在或不是有效文件: {file_path}")
        # 模拟索引过程
        logger.debug(f"正在索引文件: {path}")        
        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"文件内容长度: {len(content)} 字符")

            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            documents = document_splitter_service.split_text(content, normalized_path)
            logger.info(f"文本分割完成: {len(documents)} 个块")

            if documents:
                vector_store_manager.add_documents(documents)
                logger.info(f"向量存储完成: {len(documents)} 个向量")
            else:
                logger.warning(f"没有生成任何文档块，文件可能过小或内容不适合分割: {file_path}")


        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"读取文件失败: {e}") from e

vector_index_service = VectorIndexService()