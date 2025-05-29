from fastapi import APIRouter, Depends, HTTPException, logger, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import numpy as np
from api.schemas import ClipTextgRequest, ClipTextResponse, ClipImgRequest, ClipImgReqponse
from api.config import sk_key
from common.bot import load_jinaclipv2
from controller.clip import img2embedding, process_embedding



router = APIRouter()  # 创建子路由

security = HTTPBearer()

model = load_jinaclipv2()  # 预加载Jina的CLIP V2



@router.post("/v1/cliptext", summary="3、CLIP文字转向量", response_model=ClipTextResponse)
async def get_textembeddings(
    request: ClipTextgRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    
    
    
    # Encode text and images
    embeddings = model.encode(request.texts, normalize_embeddings=True)
  
    embeddings = [process_embedding(embedding, 1536) for embedding in embeddings]

    # Min-Max normalization
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.texts)
    total_tokens = sum(len(text) for text in request.texts)  #计算总tokens数量

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "vector"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": "jina-clip-v2",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
        "msg":"success",
    }

    return response

@router.post("/v1/clipimg", summary="4、CLIP图片转向量", response_model=ClipImgReqponse)
async def embeddings(request: ClipImgRequest,
                     credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )

    #base64图像向量化
    b64_imgs = request.b64_imgs
    img_embeddings = img2embedding(b64_imgs)
    
    
    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "vector"}
            for index, embedding in enumerate(img_embeddings)
        ],
        "model": "jina-clip-v2",
        "msg": "sucess"
    }
    return response