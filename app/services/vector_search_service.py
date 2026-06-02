"""向量检索服务"""

from typing import  Dict, Any
from app.core.milvus_client import milvus_manager
from loguru import logger
from pymilvus import Collection
from app.services.vector_embedding_service import vector_embedding_service



class SearchResult:

    def __init__(self, id: str, score: float, metadata: Dict[str, Any], content: str):
        self.id = id
        self.score = score
        self.metadata = metadata
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata,
            "content": self.content
        }
    

class VectorSearchService:

    def __init__(self):
        logger.info("VectorSearchService initialized")

    def search_similar_documents(self, query: str, top_k: int = 3):
        try:
            logger.info(f"开始搜索相似文档, 查询: {query}, topK: {top_k}")

            # 1. 将查询文本向量化
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功, 维度: {len(query_vector)}")

            # 2. 获取 collection
            collection: Collection = milvus_manager.get_collection()

            # 3. 构建搜索参数
            search_params = {
                "metric_type": "L2",  # 欧氏距离
                "params": {"nprobe": 10},
            }

            # 4. 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            # 5. 解析搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # L2 距离，越小越相似
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成, 找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e

# 全局单例
vector_search_service = VectorSearchService()
