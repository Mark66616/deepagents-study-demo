import os
from pathlib import Path
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv, find_dotenv

"""
skill使用步骤：
1、必须使用FilesystemBackend接本地文件系统
2、使用backend参数指定本地文件系统根目录
3、使用skills参数指定技能目录
4、使用checkpointer参数指定检查点保存目录
5、使用system_prompt参数指定系统提示

注意事项：
skill所在文件夹名称要和skill.md中元数据的name保持一直
skill时渐进式加载，模型只会加载skill的元数据，根据需求加载下面的提示词，所以元数据一定要简洁明了，必须那么和description
skill不是越多越好，尽量7-10个，并且要质量高不能有功能重复的skill，否则模型很容易出现摆烂或者选择困难症（注意力涣散）
skill要测试好触发的关键字是什么！！
"""

# 加载环境变量
load_dotenv(find_dotenv())

# ======================== 1. 设置 Backend ========================
# 使用 FilesystemBackend 连接到本地文件系统
# 假设 skills 目录在当前脚本同级目录下的 skills/ 中
current_dir = Path(__file__).parent.resolve()
# 我们将 root_dir 设置为当前目录 (base)
# 注意：FilesystemBackend 的 root_dir 是物理路径的根
fs_backend = FilesystemBackend(root_dir=current_dir)

# ======================== 2. 初始化 Agent ========================
llm = init_chat_model(
    model="qwen-max",
    model_provider="openai"
)

# 创建带 Skill 的 Agent
agent = create_deep_agent(
    model=llm,
    # 关键点1：注入文件系统后端
    backend=fs_backend,
    # 关键点2：告诉 Agent 在 /skills/ 目录下查找技能
    # 这里的 /skills/ 是相对于 backend root_dir 的路径
    # 物理路径为: base/skills/
    skills=["skills"],
    checkpointer=MemorySaver(),
    # System Prompt 可以很通用，具体的专业指令由 Skill 提供
    system_prompt="你是一个有用的 AI 助手。"
)

# ======================== 3. 运行演示 ========================

def run_demo():
    print("\n=== 场景：用户提供一段有问题的代码求审查 ===")
    
    bad_code = """
        def get_user(user_id):
            # 连接数据库
            import sqlite3
            conn = sqlite3.connect('test.db')
            cursor = conn.cursor()
            # 直接拼接 SQL，有注入风险！
            sql = "SELECT * FROM users WHERE id = " + user_id
            cursor.execute(sql)
            return cursor.fetchall()
    """
    
    print(f"用户代码片段:\n{bad_code}\n")
    print(">>> Agent 正在思考并匹配技能...\n")

    # 用户的提问触发了 SKILL.md 中的 description ("当用户请求进行代码审查...")
    # Agent 会自动读取 SKILL.md 的内容，并按里面的步骤执行。
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": f"请使用 code-reviewer 技能帮我 Review 一下这段代码：\n{bad_code}"}
        ],
    }, config={"configurable": {"thread_id": "skill_demo_v3"}})

    print("=== Agent 回复 (基于 code-reviewer 技能) ===")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    run_demo()

# 测试结果：
# === 场景：用户提供一段有问题的代码求审查 ===
# 用户代码片段:
#
#         def get_user(user_id):
#             # 连接数据库
#             import sqlite3
#             conn = sqlite3.connect('test.db')
#             cursor = conn.cursor()
#             # 直接拼接 SQL，有注入风险！
#             sql = "SELECT * FROM users WHERE id = " + user_id
#             cursor.execute(sql)
#             return cursor.fetchall()
#
#
# >>> Agent 正在思考并匹配技能...
#
# === Agent 回复 (基于 code-reviewer 技能) ===
# 似乎在指定的路径下找不到`code-reviewer`技能的文档。尽管如此，我可以直接对您提供的代码进行审查，并给出建议。
#
# 对于这段 Python 代码：
#
# ```python
# def get_user(user_id):
#     # 连接数据库
#     import sqlite3
#     conn = sqlite3.connect('test.db')
#     cursor = conn.cursor()
#     # 直接拼接 SQL，有注入风险！
#     sql = "SELECT * FROM users WHERE id = " + user_id
#     cursor.execute(sql)
#     return cursor.fetchall()
# ```
#
# 这里有一些问题和改进建议：
#
# 1. **SQL 注入风险**：如注释中提到的，直接将 `user_id` 拼接到 SQL 查询字符串中存在严重的 SQL 注入安全风险。应该使用参数化查询来避免这个问题。
# 2. **异常处理**：在执行数据库操作时，应当添加适当的错误处理机制，以应对可能发生的异常情况，比如连接失败或查询出错。
# 3. **资源管理**：确保在操作完成后正确关闭数据库连接和游标。可以使用 `with` 语句来自动管理这些资源。
# 4. **导入位置**：通常情况下，模块的导入应放在文件的顶部，而不是函数内部。
#
# 下面是改进后的代码示例：
#
# ```python
# import sqlite3
# from contextlib import closing
#
# def get_user(user_id):
#     try:
#         with closing(sqlite3.connect('test.db')) as conn:
#             with closing(conn.cursor()) as cursor:
#                 # 使用参数化查询防止 SQL 注入
#                 sql = "SELECT * FROM users WHERE id = ?"
#                 cursor.execute(sql, (user_id,))
#                 return cursor.fetchone()  # 假设用户 ID 是唯一的，使用 fetchone 更合适
#     except sqlite3.DatabaseError as e:
#         print(f"Database error: {e}")
#         return None
# ```
#
# 这样修改后，代码更加安全、健壮且符合 Python 的最佳实践。如果您需要进一步的帮助，请告诉我。