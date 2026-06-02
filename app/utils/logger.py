"""
日志记录
"""
from loguru import logger
def setup_logger():
    """
    配置日志系统
    """
    # 移除默认处理器
    logger.remove()
    

