"""sources：所有外部数据的唯一出入口。

依赖方向：api -> agent/services -> sources -> core。
只有这一层允许 import akshare。列名漂移也只在这里处理。
"""
