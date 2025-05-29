import base64
from PIL import Image
from io import BytesIO


def base64_to_pil_image(base64_string):
    # 移除base64编码中的前缀
    base64_string = base64_string.split(",")[-1]
    # 将base64字符串解码为字节数据
    image_data = base64.b64decode(base64_string)
    # 将字节数据转换为PIL图像
    pil_image = Image.open(BytesIO(image_data))
    #pil_image = pil_image.convert("RGB")  # 确保图像是RGB格式
    return pil_image
