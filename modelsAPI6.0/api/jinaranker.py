from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from common.logger import logger
from api.schemas import JinaRequest, JinaResponse
from api.config import sk_key
from common.bot import load_jinarerankerm0
from utils.convert import base64_to_pil_image



router = APIRouter()  # 创建子路由

security = HTTPBearer()
#yv加载Jina Reranker模型
model = load_jinarerankerm0()
model.eval()

#输入文本搜索子路由
@router.post("/v1/intextreranker",summary="5、JINA文本搜索Reranker", response_model=JinaResponse)
async def get_reranker(
    request: JinaRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    
    
    # 1. 计算文本-文本相似度
    TTscores = []
    text_pairs = [request.query, request.texts]  # 将查询和文本列表组合成对
    TTscore = model.compute_score(text_pairs, max_length=1024, doc_type="text")
    TTscores.append(TTscore)  # 确保TTscores是列表格式，即使只有一个分数
    
    # 2. 计算文本-图像相似度
    b64_imgs = request.b64_imgs
    TIscores = []
    for b64_img in b64_imgs:  # 逐个处理每个Base64字符串
        # 将单个Base64字符串转换为PIL图像
        b64_img = base64_to_pil_image(b64_img) 
   # img_embeddings = img2embedding(b64_imgs)
        image_pairs = [request.query, b64_img]
        TIscore = model.compute_score(image_pairs, max_length=2048, doc_type="image")
        TIscores.append(TIscore)
    
    # 3. 整合文本和图像结果
    all_results = []
    # 添加文本结果
    for idx, (doc, score) in enumerate(zip(request.texts, TTscores)):
        all_results.append({
            "type": "text",
            "content": doc,
            "score": score,
            "index": idx
        })
    
    # 添加图像结果
    for idx, (doc, score) in enumerate(zip(request.b64_imgs, TIscores)):
        all_results.append({
            "type": "image",
            "content": doc,  # Base64编码图像
            "score": score,
            "index": idx
        })
    
    # 4. 按相似度降序排序
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. 提取前5项
    top_items = all_results[:request.num]
    response = {
        "TOP": [top_items],
        "data": [
            all_results
        ],
        "usage": {
            "prompt_tokens": len(request.query.split()),
            "total_tokens": sum(len(doc) for doc in request.texts),
        },
        "model": "jina-reranker-m0",
        "msg": "success",
    }

    return response

#输入图片搜索
@router.post("/v1/inimgreranker",summary="6、JINA图片搜索Reranker", response_model=JinaResponse)
async def get_reranker(
    request: JinaRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        logger.error(f"密钥错误")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    
    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    # 1. 计算图片-文本相似度
    inb64_img = request.query 
    rgb_img = base64_to_pil_image(inb64_img) 
    image_pairs = [[rgb_img, doc] for doc in request.texts]
    ITscores = model.compute_score(image_pairs, max_length=2048, query_type="image", doc_type="text")
    
    # 2. 计图片-图像相似度
    b64_imgs = request.b64_imgs
    IIscores = []
    for b64_img in b64_imgs:  # 逐个处理每个Base64字符串
        # 将单个Base64字符串转换为PIL图像
        rgb_imgs = []
        b64_img = base64_to_pil_image(b64_img)
        #b64_img = b64_img.convert("RGB")  # 确保图像是RGB格式
        rgb_imgs.append(b64_img)
   # img_embeddings = img2embedding(b64_imgs)
        image_pairs = [[rgb_img, img] for img in rgb_imgs]
        IIscore = model.compute_score(image_pairs, max_length=2048, doc_type="image", query_type='image')
        IIscores.append(IIscore)
    
    # 3. 整合文本和图像结果
    all_results = []
    # 添加文本结果
    for idx, (doc, score) in enumerate(zip(request.texts, ITscores)):
        all_results.append({
            "type": "text",
            "content": doc,
            "score": score,
            "index": idx
        })
    
    # 添加图像结果
    for idx, (doc, score) in enumerate(zip(request.b64_imgs, IIscores)):
        all_results.append({
            "type": "image",
            "content": doc,  # Base64编码图像
            "score": score,
            "index": idx
        })
    
    # 4. 按相似度降序排序
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # 5. 提取前5项
    top_items = all_results[:request.num]
    response = {
        "TOP": [top_items],
        "data": [
            all_results
        ],
        "usage": {
            "prompt_tokens": len(request.query.split()),
            "total_tokens": sum(len(doc) for doc in request.texts),
        },
        "model": "jina-reranker-m0",
        "msg": "success",
    }

    return response