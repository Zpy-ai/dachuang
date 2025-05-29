from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import torch
from modelscope import AutoModel

def load_bgem3model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"本次加载模型的设备为：{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU.'}"
    )
    return SentenceTransformer("models/bge-m3", device=device)

# 预加载bge-reranker-v2-m3模型
def load_bgev2m3model():
    print(
        f"本次加载模型的设备为：{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU.'}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FlagReranker('models/bge-reranker-v2-m3', device=device, use_fp16=True)

def load_jinarerankerm0():
    print(
        f"本次加载模型的设备为：{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU.'}"
    )
    return AutoModel.from_pretrained(
        'models/jina-reranker-m0',
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to('cuda')  # or 'cpu' if no GPU is available


clip_model = None
device = None

def init_model():
    """初始化模型（仅需调用一次）"""
    global clip_model, device
    print(f"本次加载模型的设备为：{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU.'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = SentenceTransformer('models/jina-clip-v2', trust_remote_code=True, truncate_dim=1024)
    clip_model = clip_model.to(device)
    print(f"模型已加载到设备: {device}")

def load_jinaclipv2():
    """获取已初始化的模型"""
    if clip_model is None:
        init_model()
    return clip_model
