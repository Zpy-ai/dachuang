from fastapi import APIRouter, Depends, HTTPException, logger, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
#from common.logger import logger
from controller.reranker import sort_items
from api.schemas import BgeRerankeRequest, BgeRerankeResponse
from api.config import sk_key
from common.bot import load_bgev2m3model

router = APIRouter()  # 创建子路由

security = HTTPBearer()

reranker = load_bgev2m3model()  # 预加载BGE模型

@router.post("/v1/bgereranker",summary="2、BGE文字转向量Reranker", response_model=BgeRerankeResponse)
async def get_reranker(
    request: BgeRerankeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    
    query_doc_pairs = [[request.query, doc] for doc in request.texts]
    
    # You can map the scores into 0-1 by set "normalize=True"
    scores = reranker.compute_score(query_doc_pairs, normalize=True)

    top_items = sort_items(request.texts, scores)# 排序并获取结果
    top_items = top_items[:request.num]
    response = {
        "TOP": [top_items],
        "data": [
            {"score": score, "index": index, "object": "score"}
            for index, score in enumerate(scores)
        ],
        "model": "bge-reranker-v2-m3",
        "msg":"success",
    }

    return response