"""services：把 sources + analysis 组合成具体用例。

这一层的函数接收显式依赖（repo / settings），不读任何模块级全局变量。
本文件刻意不做 re-export，避免包初始化时的循环导入。
"""
