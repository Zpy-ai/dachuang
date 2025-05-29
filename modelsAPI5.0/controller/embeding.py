import numpy as np
from sklearn.preprocessing import PolynomialFeatures



# 功能函数
def process_embedding(embedding, target_length):
    """扩展或截断嵌入向量以匹配目标长度。"""
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1)).flatten()

    if len(expanded_embedding) > target_length:
        return expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        return np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    return expanded_embedding
