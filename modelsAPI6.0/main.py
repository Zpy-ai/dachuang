from fastapi import FastAPI
from utils.gc import torch_gc
from api.router import api_router
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pass
    except:
        pass
    yield
    try:
        torch_gc()  # 释放显存
    except:
        pass

app = FastAPI(lifespan=lifespan)  # 创建一个FastAPI应用，屏蔽接口文档
# app = FastAPI(lifespan=lifespan, docs_url=None,
#               redoc_url=None)  # 创建一个FastAPI应用，屏蔽接口文档

app.include_router(api_router)  # 将路由添加到主程序中
# 配置允许域名列表、允许方法、请求头、cookie等
app.add_middleware(
    CORSMiddleware,   # 中间件，用于跨域
    allow_origins=["*"],  # 允许域名
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许的方法
    allow_headers=["*"],  # 允许的请求头
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=6008, reload=False)
