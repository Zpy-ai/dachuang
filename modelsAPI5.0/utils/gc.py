import torch


def torch_gc():  # 回收显存
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        for gpu_index in range(gpu_count):
            with torch.cuda.device(f"cuda:{gpu_index}"):  # 指定CUDA设备
                torch.cuda.empty_cache()  # 清空CUDA缓存
                torch.cuda.ipc_collect()  # 收集CUDA内存碎片
