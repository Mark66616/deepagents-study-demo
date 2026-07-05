""""
Middleware 是 DeepAgents 的「流程拦截器」，可以在 Agent 执行的 关键生命周期节点 （如工具调用前 / 后、思考完成后、回复生成前）插入自定义逻辑，实现：

- 操作日志记录
- 权限校验 / 参数过滤
- 工具调用结果修改
- 异常捕获 / 兜底处理
- 自定义监控 / 告警

官网文档：
https://docs.langchain.com/oss/python/deepagents/customization#middleware
https://reference.langchain.com/python/langchain/middleware
提供了很多常用的中间件实现，比如：
SummarizationMiddleware：可以对工具调用进行总结（指定token（上下文）的压缩阈值（通常为模型上下文窗口的三分之二）以及对话轮数阈值），帮忙提取上下文摘要，进行压缩
Model Call Limit Middleware：可以指定线程id限制工具在一次回话（一次invoke执行）或者一个线程调用模型的次数，可以指定exit_behavior参数为end-退出或者error-抛异常
Tool call limit middleware: 可以指定线程id调用工具一次回话（一次invoke执行）或者一个线程调用的次数、
HumanInTheLoopMiddleware：可以实现人类在回路中的中间件
"""
# -*- coding: utf-8 -*-
"""
DeepAgents Middleware 极简案例
核心：实现工具调用的日志监控中间件
"""
import os
import time

from langchain.agents.middleware import wrap_tool_call, ToolCallLimitMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())


# ======================== 1. 定义测试工具 ========================
@tool
def add_numbers(a: int, b: int):
    """计算两个数字的和"""
    time.sleep(0.5)  # 模拟耗时操作
    result = a + b
    print(f"[工具执行] {a} + {b} = {result}")
    return result


@wrap_tool_call
def log_tool_call(request, handler):
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    # 1. 前置逻辑
    print(f"\n[前置中间件] 工具调用开始 - 工具名: {tool_name}, 参数: {tool_args}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # 2. 执行工具 (调用 handler)
    result = handler(request)

    end_time = time.time()
    duration = end_time - start_time

    # 3. 后置逻辑
    # 先尝试从 result 对象中获取 content 属性；如果 result 没有 content 属性（比如不是 ToolMessage 类型），就把 result 转成字符串作为兜底值。
    content = getattr(result, "content", str(result))
    print(f"[后置中间件] 工具调用完成 - 工具名: {tool_name}, 结果: {content}, 耗时: {duration:.2f}秒")

    return result

# ======================== 3. 配置Agent并绑定Middleware ========================
# 初始化LLM
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 创建Agent，绑定中间件
deep_agent = create_deep_agent(
    model=llm,
    tools=[add_numbers],
    checkpointer=InMemorySaver(),
    # 绑定中间件：传入 Middleware 实例列表
    middleware=[log_tool_call
                ,ToolCallLimitMiddleware(tool_name="add_numbers",
                                         thread_limit=2,
                                         run_limit=2,
                                         exit_behavior="error")
                ,ModelCallLimitMiddleware(thread_limit=1,
                                          run_limit=1,
                                          exit_behavior="end")],
    system_prompt="你是一个计算器助手，使用add_numbers工具完成加法计算，回答仅返回计算结果。"
)

# ======================== 4. 执行测试 ========================
if __name__ == "__main__":
    # 会话配置
    thread_config = {"configurable": {"thread_id": "middleware_test_1"}}

    # 调用Agent
    result = deep_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "帮我计算 100 + 200 的结果，然后在计算123+345的结果"}
            ]
        },
        config=thread_config
    )

    # 输出最终结果
    print("\n=== 最终回复 ===")
    print(result["messages"][-1].content)

# 提示词为：“帮我计算 100 + 200 的结果，然后在计算123+345的结果”，超出ToolCallLimitMiddleware限制（均为1），直接返回超限，end策略没有调用工具，error会直接抛异常
# === 最终回复 ===
# 'add_numbers' tool call limit reached: thread limit exceeded (2/1 calls) and run limit exceeded (2/1 calls).

# 同样上述提示词，如果ModelCallLimit超出限制，会调用工具，并且执行对应的exit_behavior
# [前置中间件] 工具调用开始 - 工具名: add_numbers, 参数: {'a': 100, 'b': 200}, 时间: 2026-07-05 13:03:35
#
# [前置中间件] 工具调用开始 - 工具名: add_numbers, 参数: {'a': 123, 'b': 345}, 时间: 2026-07-05 13:03:35
# Traceback (most recent call last):
#   File "E:\dev_space\pycharm\deepagents-study-demo\base\deepagents_middleware_and_custom_middleware.py", line 102, in <module>
#     result = deep_agent.invoke(
#              ^^^^^^^^^^^^^^^^^^
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\pregel\main.py", line 3928, in invoke
# [工具执行] 100 + 200 = 300
# [后置中间件] 工具调用完成 - 工具名: add_numbers, 结果: 300, 耗时: 0.50秒
# [工具执行] 123 + 345 = 468
# [后置中间件] 工具调用完成 - 工具名: add_numbers, 结果: 468, 耗时: 0.50秒
#     for chunk in self.stream(
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\pregel\main.py", line 2982, in stream
#     for _ in runner.tick(
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\pregel\_runner.py", line 207, in tick
#     run_with_retry(
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\pregel\_retry.py", line 617, in run_with_retry
#     return task.proc.invoke(task.input, config)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 684, in invoke
#     input = context.run(step.invoke, input, config, **kwargs)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 426, in invoke
#     ret = self.func(*args, **kwargs)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\dev_space\pycharm\deepagents-study-demo\.venv\Lib\site-packages\langchain\agents\middleware\model_call_limit.py", line 195, in before_model
#     raise ModelCallLimitExceededError(
# langchain.agents.middleware.model_call_limit.ModelCallLimitExceededError: Model call limits exceeded: thread limit (1/1), run limit (1/1)
# During task with name 'ModelCallLimitMiddleware.before_model' and id '2a60cd5f-2355-7e89-73be-30e9fd167a32'