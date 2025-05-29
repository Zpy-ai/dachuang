from fastapi import APIRouter
from api import embeding, bgeranker, clip, health, jinaranker

api_router = APIRouter(prefix="/api")  # 总路由

api_router.include_router(router=health.router) # 健康检查接口
api_router.include_router(router=embeding.router)  # bgeEmbedding接口
api_router.include_router(router=bgeranker.router)  # bge ReRanker接口
api_router.include_router(router=clip.router)  # Clip相关接口
api_router.include_router(router=jinaranker.router) # Jina Reranker接口

