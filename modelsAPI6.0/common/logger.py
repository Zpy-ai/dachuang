import os
import loguru


def create_log(log_filepath, ProName):  # 程序日志设置
    # loguru.logger.remove(handler_id=None)  # 不在控制台输出信息
    # 设置日志格式，日志最长保留时间位3天，日志大小超过10MB则压缩为zip文件，输出文件格式为utf-8，开启多进程安全。
    loguru.logger.add(log_filepath, rotation="10 MB", retention="3 day", compression="zip",
                      encoding="utf-8", enqueue=True, filter=lambda record: record["extra"]["name"] == ProName)
    logger = loguru.logger.bind(name=ProName)  # 将日志绑定到对应的变量中，从而实现多文件输出日志
    return logger  # 返回日志对象


logger = create_log(
    # 程序运行日志
    log_filepath=f"modelsAPI/logs/runtime/{os.getenv('SERVICE_NAME', 'modelsAPI').lower()}.log", ProName=os.getenv('SERVICE_NAME', "modelsAPI"))
