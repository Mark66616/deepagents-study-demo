from pathlib import Path  # 导入Path类
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# 1. 准备本地工作目录（用Path改写）
workspace_dir = Path("./agent_workspace").resolve()  # resolve() 等价于 os.path.abspath()，获取绝对路径
if not workspace_dir.exists():  # 等价于 os.path.exists()
    workspace_dir.mkdir(parents=True, exist_ok=True)  # 等价于 os.makedirs()

print(f"Agent 的工作目录已设置为: {workspace_dir}")

# 2. 配置本地文件系统后端
# virtual_mode=True 开启安全沙箱模式，限制 Agent 只能访问 workspace_dir
# 如果不开启，Agent 将有访问整个文件系统的权限，并且会随意写文件
backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# 3. 创建 Agent
# System Prompt 提示 Agent 按需创建文件
agent = create_deep_agent(
    model=llm,
    backend=backend,
    system_prompt="你是一个智能助手。你可以使用文件工具来读写文件，但只有在用户明确要求时才创建文件。"
)

# 4. 运行并验证
print("\n=== Case 1: 普通问答（不应该生成文件） ===")
result1 = agent.invoke({"messages": [{"role": "user", "content": "请告诉我 Python 是什么时候发明的？"}]})
print("Agent 回复:", result1["messages"][-1].content)

# === Case 1: 验证没有生成文件 ===
# 替换 os.listdir → Path.iterdir() + 判断是否有文件
files = list(workspace_dir.iterdir())  # 列出目录下所有条目（文件/子目录）
if not files:
    print("Case 1 成功：目录下没有生成任何文件。")
else:
    # 提取文件名/目录名，保持输出格式一致
    file_names = [f.name for f in files]
    print(f"Case 1 失败：目录下生成了文件: {file_names}")

print("\n=== Case 2: 明确要求生成文件 ===")
result2 = agent.invoke({"messages": [{"role": "user", "content": "帮我写一份关于 Java 的简短介绍，并保存为 java_intro.md"}]})
print("Agent 回复:", result2["messages"][-1].content)

# 5. 验证 Case 2 文件是否真实存在
# 替换 os.path.join → Path / 运算符
file_path = workspace_dir / "java_intro.md"  # Path 拼接更直观
if file_path.exists():  # 替换 os.path.exists → Path.exists()
    print(f"\nCase 2 成功！文件已生成在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:  # Path 对象可直接传入 open
        print(f" 文件内容预览:\n{f.read()[:100]}...")
else:
    print("\nCase 2 失败：文件未生成。")