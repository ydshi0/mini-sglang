# 旧（错误被吞掉，只报 "unknown location"）
from .spec_llm import SpeculativeLLM

# 新（底层 import 失败时会显示真正的 traceback）
def __getattr__(name):
    if name == "SpeculativeLLM":
        from .spec_llm import SpeculativeLLM
        return SpeculativeLLM
    raise AttributeError(...)