import os
from typing import TypedDict, Annotated

from deepagents import create_deep_agent, CompiledSubAgent
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 加载环境变量
load_dotenv(find_dotenv())

# deepagent兼容langchain的方法
# 1.graph中定义的state必须包含 messages属性，
# 2.messages这个属性必须是Annotated[list[AnyMessage], add_messages]，其中add_messages是langgraph提供的reducer函数，不添加新消息可能会覆盖历史消息，导致对话上下文丢失
# 3.然后使用CompiledSubAgent将graph封装为CompiledSubAgent
# 4.CompiledSubAgent就可以作为subagent了

# --- 1. 定义子智能体 (基于 StateGraph) ---

# 定义 State (必须包含 messages)
class SubState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 定义节点逻辑 (增加打印语句，证明它被触发了)
def processing_node(state: SubState):
    print("\n    >>> [子智能体内部] 收到任务，正在处理...")

    # 获取主智能体传来的任务描述
    last_msg = state["messages"][-1]
    print(f"    >>> [子智能体内部] 输入内容: {last_msg.content}")

    # 模拟处理逻辑
    result_text = f"【已由Graph处理】经核查，业务逻辑处理完毕。原始内容：{last_msg.content}"

    print(f"    >>> [子智能体内部] 处理完成，准备返回。\n")
    return {"messages": [AIMessage(content=result_text)]}


# 构建图
workflow = StateGraph(SubState)
workflow.add_node("worker", processing_node)
workflow.set_entry_point("worker")
workflow.add_edge("worker", END)
compiled_graph = workflow.compile()

# 封装为 CompiledSubAgent
sub_agent_config = CompiledSubAgent(
    name="complex_worker",
    description="处理复杂业务逻辑、核查任务的子智能体。当用户提到'复杂业务'或'核查'时调用。",
    runnable=compiled_graph
)

# --- 2. 创建主智能体 ---

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

deep_agent = create_deep_agent(
    model=llm,
    subagents=[sub_agent_config],
    system_prompt="你是一个协调员。遇到复杂任务时，必须调用 complex_worker 子智能体处理。"
)

# --- 3. 运行测试 (带漂亮的日志) ---

if __name__ == "__main__":
    query = "请帮我处理这个复杂业务：核对用户 ID 9527 的数据。"
    print(f"User: {query}")
    print("=" * 60)

    # 使用 stream 并解析结果
    for chunk in deep_agent.stream({"messages": [HumanMessage(content=query)]}):
        print(f"chunk结果：: {chunk}")