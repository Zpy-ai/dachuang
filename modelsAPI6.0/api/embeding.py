from fastapi import APIRouter, logger
from api.config import sk_key
from api.schemas import BgeEmbeddingRequest, BgeEmbeddingResponse
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from controller.embeding import process_embedding
from common.bot import load_bgem3model
import numpy as np



router = APIRouter()  # 创建子路由

security = HTTPBearer()

model = load_bgem3model()  # 预加载BGE模型

@router.post("/v1/bgeembedding",summary="1、BGE文本转向量", response_model=BgeEmbeddingResponse)
async def get_embeddings(
    request: BgeEmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )

    # 计算嵌入向量
    embeddings = [
        model.encode(text, normalize_embeddings=True) for text in request.texts
    ]
    embeddings = [process_embedding(embedding, 1024) for embedding in embeddings]

    # Min-Max normalization
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 转换为列表
    prompt_tokens = sum(len(text.split()) for text in request.texts)
    total_tokens = sum(len(text) for text in request.texts)  # 示例：计算总tokens数量

    response = {
        "data": [
            {"embedding": embedding.tolist(), "index": index, "object": "vector"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": "bge-m3",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
        "msg": "success",
    }

    return response
