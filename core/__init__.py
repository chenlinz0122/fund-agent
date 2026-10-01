"""core：与业务无关的基础设施（缓存、限流、熔断、异常）。

这一层不 import akshare、不 import pandas 业务逻辑、不 import FastAPI 路由。
"""
