from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
import os
from dotenv import load_dotenv, find_dotenv

"""
子agent-dict方式的创建参数
# | 字段名        | 类型                 | 必填 / 可选 | 核心描述                                                     | 继承规则（与主代理的关系）               |
# | :------------ | :------------------- | :---------- | :----------------------------------------------------------- | :--------------------------------------- |
# | name          | str                  | 必填        | 子代理的唯一标识；主代理调用 `task()` 工具时会使用该名称，也会作为 AIMessage / 流式输出的元数据，用于区分不同代理 | -（无继承，需自定义）                    |
# | description   | str                  | 必填        | 子代理的职能描述（需具体、以行动为导向）；主代理会根据此信息判断是否将任务委派给该子代理 | -（无继承，需自定义）                    |
# | system_prompt | str                  | 可选        | 子代理的执行指令，需包含工具使用指导、输出格式要求等核心规则 | 不继承主代理的，需自定义                 |
# | tools         | list[Callable]       | 可选        | 子代理可使用的工具列表；建议极简配置，仅保留必要工具         | 不继承主代理的，需自定义                 |
# | model         | str \| BaseChatModel | 可选        | 子代理使用的模型：1. 传字符串（如 `openai:gpt-5`）2. 传 LangChain 模型对象（如 `init_chat_model("gpt-5")`）省略则使用主代理的模型 | 默认继承主代理的模型，自定义会覆盖默认值 |
# | middleware    | list[Middleware]     | 可选        | 自定义中间件，用于实现日志记录、速率限制、自定义行为等功能   | 不继承主代理的，需自定义                 |
# | interrupt_on  | dict[str, bool]      | 可选        | 为特定工具配置 “人机协作流程（HITL）”；需搭配检查点（checkpointer）使用 | -                                        |
# | skills        | list[str]            | 可选        | 技能文件的来源路径（如 `["/skills/research/"]`），用于加载子代理专属技能 | -                                        |

注意上述dict类型定义的时候，除了model之外子agent可以继承主agent（主定义子不定义默认使用相同的模型），其他的都是子agent独立的。
这样设计的好处：
- 专注 ：防止上下文污染。比如负责写代码的 Agent 不需要知道负责写文案的 Agent 的具体指令。
- 安全 ：限制工具权限。比如只有顶层 Agent 能批准发布，底层 Agent 只能提交代码。
- 模块化 ：方便独立测试和复用子 Agent。

创建主agent：
main_agent = create_deep_agent(
    model=llm,
    tools=[],  # 主智能体本身不带工具，依靠子智能体
    subagents=[weather_agent, math_agent, translate_agent],
    system_prompt="你是一个全能管家。你会根据用户的需求，调度不同的助手来解决问题。"
)

subagents是一个字典列表，主agent通过子agent的description的定义来决定何时调用该agent。
当主智能体发现用户意图匹配某个子智能体的 `description` 时，会自动生成一个 `task` 工具调用（AIMessage中toolCalls中的那么为task），将任务分发下去。

另外deepagent的steam和invoke都是支持异步调用的，但是要调用对应的异步方法，如astream或者ainvoke等。
"""

load_dotenv(find_dotenv())

# 极简初始化（自动读取OPENAI环境变量）
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    temperature=0.1,  # 自定义温度（更严谨的回答）
    model_provider="openai"
)

# 1. 定义子智能体：天气助手
weather_agent = {
    "name": "weather_helper",
    "description": "用于查询天气信息。当用户询问天气时，请调用此助手。",
    "system_prompt": "你是一个天气助手。无论用户问哪个城市的天气，你都统一回答：'今日天气晴朗，气温 25 度，适合出游。'",
    "tools": []  # 这里不需要额外工具，仅靠 prompt 回复
}

# 2. 定义子智能体：计算助手
math_agent = {
    "name": "math_helper",
    "description": "用于处理数学计算问题。",
    "system_prompt": "你是一个严谨的数学助手。请帮助用户计算数学问题。",
    "tools": []
}

# 3. 定义子智能体：翻译助手
translate_agent = {
    "name": "translator",
    "description": "用于中英互译任务。",
    "system_prompt": "你是一个翻译助手。如果是中文请翻译成英文，如果是英文请翻译成中文。",
    "tools": []
}

# 4. 创建主智能体，并注册子智能体
main_agent = create_deep_agent(
    model=llm,
    tools=[],  # 主智能体本身不带工具，依靠子智能体
    subagents=[weather_agent, math_agent, translate_agent],
    system_prompt="你是一个全能管家。你会根据用户的需求，调度不同的助手来解决问题。"
)


# 5. 可视化运行 (Stream)
# 使用 stream() 替代 invoke()，可以实时打印出智能体的“调度”过程，看到它如何分发任务
def test_stream(query):
    print(f"\n>>> 提问: {query}")
    # 遍历流式输出
    for chunk in main_agent.stream({"messages": [{"role": "user", "content": query}]}):
        # chunk 是一个字典，键是节点名 (如 'model', 'tools')，值是该节点的状态更新
        for node_name, state in chunk.items():
            if not state or "messages" not in state: continue
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                # 1. 模型节点 (model)：决定下一步行动
                if node_name == "model":
                    # 如果有 tool_calls，说明模型决定调用工具或子智能体
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            if tool_call['name'] == 'task':
                                sub_agent = tool_call['args'].get('subagent_type')
                                print(f"[模型决策] 呼叫子智能体: {sub_agent}")
                            else:
                                print(f"[模型决策] 调用工具: {tool_call['name']},参数为：{tool_call['args']}")
                    # 如果没有 tool_calls 且有 content，说明是最终回复
                    elif last_msg.content:
                        print(f"[最终回复] {last_msg.content}")

                # 2. 工具节点 (tools)：显示工具/子智能体的执行结果
                elif node_name == "tools":
                    content_preview = ''
                    if len(last_msg.content) > 100:
                        # 取前100个字符 + 省略号（截断预览）
                        content_preview = last_msg.content[:100] + "..."
                    else:
                        # 内容较短，直接完整显示
                        content_preview = last_msg.content
                    print(f"[执行结果] {content_preview}")
test_stream("北京今天天气怎么样？")
test_stream("100 + 256 等于多少？")