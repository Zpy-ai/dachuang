import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from utils.convert import base64_to_pil_image
import torch
from common.bot import load_jinaclipv2
from common.logger import logger



model = load_jinaclipv2()  # 预加载Jina的CLIP V2
# 设定设备，确保模型在正确的设备上运行  

def img2embedding(b64_imgs):  # 将图片转换为向量 

    try:
        with torch.no_grad():
            embeddings = []
            for b64_img in b64_imgs:
                image = base64_to_pil_image(b64_img)
                image_features = model.encode(image, convert_to_tensor=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.tolist())
            return embeddings
    except Exception as e:
            logger.error(f"clip文字转向量失败:{e}")

# text功能函数
def process_embedding(embedding, target_length):
    """扩展或截断嵌入向量以匹配目标长度。"""
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1)).flatten()

    if len(expanded_embedding) > target_length:
        return expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        return np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    return expanded_embedding