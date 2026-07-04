from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, FilesystemBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
import os
from pathlib import Path  # 新增导入 Path 类

load_dotenv(find_dotenv())

# 1. 准备 Store
store = InMemoryStore()

# 2. 配置 LLM
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 3. 定义混合后端工厂函数
def create_composite_backend(runtime):
    # 后端 A: 本地文件系统 (存普通文件)
    # 1. 准备本地工作目录（用Path改写）
    workspace_dir = Path("./agent_workspace").resolve()  # resolve() 等价于 os.path.abspath()，获取绝对路径
    if not workspace_dir.exists():  # 等价于 os.path.exists()
        workspace_dir.mkdir(parents=True, exist_ok=True)  # 等价于 os.makedirs()
    fs_backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)

    # 后端 B: 数据库存储 (存重要记忆)
    store_backend = StoreBackend(runtime)

    # 组合后端: 配置路由规则
    return CompositeBackend(
        default=fs_backend,  # 默认走本地文件系统
        routes={
            "/store/": store_backend  # 以 /store/ 开头的路径走数据库存储
        }
    )


agent = create_deep_agent(
    model=llm,
    store=store,
    backend=create_composite_backend,  # 传入工厂函数
    tools=[],
    system_prompt="""你是一个智能助手。
    - 普通文件：直接写入文件名（如 `report.txt`），保存到本地 workspace。
    - 重要记忆：写入 `/store/` 目录（如 `/store/profile.txt`），保存到store指定的存储方式中。
    """
)

# 4. 运行 Agent
print("\n=== 测试混合存储 ===")
config = {"configurable": {"thread_id": "thread_composite"}}

# 任务：同时触发两种存储路径
user_input = "1. 创建本地文件 local.txt，内容'本地文件'。\n2. 创建记忆文件 /store/memory.txt，内容'重要记忆'。"
print(f"用户指令: {user_input}")

result = agent.invoke({
    "messages": [{"role": "user", "content": user_input}]
}, config=config)

print("Agent 回复:", result["messages"][-1].content)

# 5. 验证结果
print("\n=== 验证本地文件 (Filesystem) ===")
# 替换 os.path.join + os.path.exists 为 Path 写法
local_path = Path("agent_workspace") / "local.txt"  # Path 拼接路径
if local_path.exists():  # Path 内置 exists 方法
    print(f"本地文件存在: {local_path}")
else:
    print("本地文件缺失")

print("\n=== 验证数据库存储 (Store) ===")
# CompositeBackend 会自动剥离路由前缀，所以 /store/memory.txt 在 Store 中的 Key 为 /memory.txt
items = store.search(("filesystem",))
for item in items:
    print(item)

# === 测试混合存储 ===
# 用户指令: 1. 创建本地文件 local.txt，内容'本地文件'。
# 2. 创建记忆文件 /store/memory.txt，内容'重要记忆'。
# Agent 回复: 两个文件已创建完成：
#
# - `local.txt` — 内容：`本地文件`（本地 workspace）
# - `/store/memory.txt` — 内容：`重要记忆`（store 存储）
#
# === 验证本地文件 (Filesystem) ===
# 本地文件存在: agent_workspace\local.txt
#
# === 验证数据库存储 (Store) ===
# Item(namespace=['filesystem'], key='/memory.txt', value={'content': '重要记忆', 'encoding': 'utf-8', 'created_at': '2026-07-04T15:19:14.800695+00:00', 'modified_at': '2026-07-04T15:19:14.800695+00:00'}, created_at='2026-07-04T15:19:14.800695+00:00', updated_at='2026-07-04T15:19:14.800695+00:00', score=None)