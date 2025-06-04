# 基于FastAPI的本地大模型接口开发
## 一、简介
一个基于FastAPI的大模型接口项目，它集成文本与多模态大模型，支持批量文本、base64图片输入，返回向量数据、相关度排序、模型信息及taoken使用统计；配置CORS中间件允许全来源跨域请求，设置请求头密钥增强安全性。
## 二、项目结构
![@70ADS$YP865GPBKII@`X{B](https://github.com/user-attachments/assets/13f695a0-31ae-4359-b8be-32bf81f967ee)


## 三、环境配置
### 3.1 依赖安装
项目依赖的 Python 包列表在requirements.txt文件中，可使用以下命令安装所有依赖：
pip install -r requirements.txt

### 3.2 API 密钥配置
在api/config.py文件中配置 API 密钥sk-key，默认值为sk-proj-mimouse，请根据实际需求修改。
from os import getenv

sk_key = getenv("sk-key", "sk-proj-mimouse")

## 四、使用指南
### 4.1 启动项目
在项目根目录下，运行main.py文件。
### 4.2 API 接口说明
健康检查接口：http://127.0.0.1:6008/api/v1/apihealth，用于检查项目是否正常启动，返回{"status": "ok"}表示项目运行正常。

bge-m3接口：/api/v1/embedding，这个接口调用的是文本转向量的模型。

bge-reranker-v2-m3接口：/api/v2/reranker，这个模型可以根据文本筛选出与目标文本更匹配的重排模型。

jina-clip-v2接口本模型是将文本和图像转换为向量的模型。我为其分配了两个路由，如下：

clip文本转换向量接口:/api/v2/clip/text;

clip图片转换向量接口（base64）:/api/v2/clip/img,图片格式为base64。

jina-reranker-m0接口

这个模型是根据文本和图像筛选出与目标文本或图片更匹配的重排模型。分为两个接口，一是文本输入接口，计算与当前输入文本的相关度，并返回相关度最高的五条文本或图片，二是图片输入接口，计算与当前输入图片的相关度，并返回相关度最高的五条文本或图片。

文本输入接口:/api/v1/text/reranker,该接口query为文本，实现文本—文本和文本—图像相关度计算。

图片输入接口（base64）:/api/v1/img/reranker,该接口query为base64图片，实现图片—图片和图片—文本相关度计算。

# 4.3 请求示例
以 BGE 文本嵌入接口为例，使用curl命令发送请求：
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/embedding' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer sk-proj-mimouse' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["示例文本1", "示例文本2"],
  "model": "bge-m3"
}'


