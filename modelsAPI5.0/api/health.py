from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["一、通用接口"])  # 创建子路由


# 健康检查接口，检查项目是否正常启动
@router.get("/health", summary="健康检查")
def health():
    return JSONResponse(
        content={"msg": "OK"},
        status_code=200
    )
