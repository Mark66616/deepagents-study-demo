from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, StateBackend
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv(find_dotenv())

# 生产环境建议使用 RedisStore: from langgraph.store.redis import RedisStore

# 1. 准备 Store (模拟数据库)
# InMemoryStore 是轻量级内存存储，重启后数据丢失。
store = InMemoryStore()

# 2. 配置 Store 后端
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# StoreBackend 将 Agent 的文件操作转换为对 Store 的读写
# 默认情况下，它将文件存储在 ("filesystem", filename) 的 Key 下
agent = create_deep_agent(
    model=llm,
    backend=StoreBackend(store=store), # 注意：此处主 Backend 设为 StateBackend，并且传入存储的数据库配置，StoreBackend 通常作为辅助或通过 Composite 使用
    tools=[],
    system_prompt="请把用户的重要信息保存到 user_profile.txt"
)

# 注意：为了让 Agent 直接使用 StoreBackend 存储文件，
# 通常我们会直接将 backend 设置为 StoreBackend，或者在 CompositeBackend 中路由。
# 下面的示例主要演示 Store 的跨线程数据读取能力。

# 3. 运行 Agent (Thread A) - 写入记忆
print("\n=== 写入记忆 (Thread A) ===")
config_a = {"configurable": {"thread_id": "thread_a"}}
# 假设 Agent 内部逻辑会将信息写入 Store（需配合正确的 Backend 配置，此处简化演示 Store 交互）
# 在实际使用 StoreBackend 时，Agent 调用 write_file("user_profile.txt") 会被存入 Store
result = agent.invoke({
    "messages": [{"role": "user", "content": "我叫大风子，我的幸运数字是 7。"}]
}, config=config_a)

print("Agent 回复:", result["messages"][-1].content)

# 4. 运行 Agent (Thread B) - 跨线程读取
print("\n=== 读取记忆 (Thread B) ===")
# 使用不同的 thread_id，模拟另一个会话
config_b = {"configurable": {"thread_id": "thread_b"}} # 注意这里是 thread_b

# 这里的关键点是：Store 是共享的。Thread B 可以读取 Thread A 写入的数据。
result_b = agent.invoke({
    "messages": [{"role": "user", "content": "请读取 user_profile.txt 告诉我，我叫什么名字？我的幸运数字是多少？"}]
}, config=config_b)

print("Agent (Thread B) 回复:", result_b["messages"][-1].content)

# 验证：直接检查 Store 数据
print("\n=== 验证 Store 数据 ===")
# 所有文件的创建、修改、读取操作，都会自动关联 ("filesystem",) 顶级命名空间；
items = store.search(("filesystem",))
for item in items:
    print(f"Key: {item.key}")
    print(f"Value: {item.value}")