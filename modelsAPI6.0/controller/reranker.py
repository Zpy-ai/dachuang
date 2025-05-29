from typing import List
from fastapi import logger


def sort_items(texts: List[str], scores: List[float]) -> List[dict]:
    """
    按分数对文本进行排序的函数
    
    参数:
        texts: 文本列表（与scores一一对应）
        scores: 分数列表（需与texts长度一致）
    
    返回:
        排序后的列表，每个元素包含document、score和原始index
    """
    try:
        # 生成带索引的字典列表并排序
        ranked_items = sorted(
            [{"document": doc, "score": score, "index": idx} 
            for idx, (doc, score) in enumerate(zip(texts, scores))],
            key=lambda x: x["score"],
            reverse=True
        )
        return ranked_items  # 返回前5个结果
    except Exception as e:
            logger.error(f"jinareranker重排错误:{e}")
