import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient

# 加载环境变量（使用 find_dotenv() 自动查找 .env 文件，无论你在哪个目录下运行脚本都能正确加载环境变量）
load_dotenv(find_dotenv())

# 创建tavily_client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 定义搜索工具
@tool
def internet_search(query: str,
                    max_results: int = 5,
                    topic: Literal["general", "news", "finance"] = "general",
                    include_raw_content: bool = False):
    """
    使用Tavily搜索
    :param query: 搜索关键字
    :param max_results: 返回结果数量
    :param topic: 主题类型
    :param include_raw_content: 是否返回详细结果
    :return: 搜索结果列表
    """
    return tavily_client.search(query=query,
                                max_results=max_results,
                                topic=topic,
                                include_raw_content=include_raw_content)


# 极简初始化（自动读取OPENAI环境变量）
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai"
)

# api地址 https://reference.langchain.com/python/deepagents/graph/
# 功能等价于langchain的 create_agent
deep_agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    subagents=[],
    system_prompt="""
      你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。
    """
)

# 运行代理
prompt = input("输入你关心的问题！")
stream = deep_agent.stream({
    "messages": [
        {"role": "user", "content": f"{prompt}"}
    ]
})

for chunk in stream:
    """
        # 场景1：单个节点更新（常见）
        chunk = {
            "model": {"messages": [AIMessage(content='', tool_calls=[...])]}  # 仅模型节点更新
        }

        # 场景2：多个节点同时更新（少数但存在）
        chunk = {
            "model": {"messages": [AIMessage(content='最终回复...')]},  # 模型节点
            "tools": {"messages": [ToolMessage(content='工具结果...')]},  # 工具节点
            "todos": {"todos_list": ["已完成：搜索宇树机器人新闻"]}       # 待办节点
        }
    """
    for node_name, state in chunk.items():
        print(f"本次处理的节点类型{node_name}")
        # 有些中间节点 例如：TodoListMiddleware 没有 messages跳过！
        if not state or "messages" not in state: continue
        # 有的直接获取
        messages = state["messages"]
        # message不为null!并且是集合类型
        if messages and isinstance(messages, list):
            # 获取最后一条就是最终结果
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
                    print(f"📝 [最终回复] {last_msg.content}")
            # 2. 工具节点 (tools)：显示工具/子智能体的执行结果
            elif node_name == "tools":
                # ToolMessage 的 content 是工具返回的原始数据 (可能是 JSON 字符串)
                # 建议只打印前 100 个字符，避免刷屏
                # 展开成普通if-else，更易理解
                content_preview = ''
                if len(last_msg.content) > 100:
                    # 取前100个字符 + 省略号（截断预览）
                    content_preview = last_msg.content[:100] + "..."
                else:
                    # 内容较短，直接完整显示
                    content_preview = last_msg.content
                print(f"[执行结果] {content_preview}")

# 输入你关心的问题！人工智能和机器人的热点新闻
# 本次处理的节点类型PatchToolCallsMiddleware.before_agent
# 本次处理的节点类型model
# [模型决策] 调用工具: internet_search,参数为：{'query': '人工智能 最新热点新闻 2025', 'topic': 'news', 'max_results': 8, 'include_raw_content': True}
# [模型决策] 调用工具: internet_search,参数为：{'query': '机器人 最新热点新闻 2025', 'topic': 'news', 'max_results': 8, 'include_raw_content': True}
# 本次处理的节点类型TodoListMiddleware.after_model
# 本次处理的节点类型tools
# [执行结果] Tool result too large, the result of this tool call call_6586652948914a0aaaa1d8c3 was saved in the f...
# 本次处理的节点类型tools
# [执行结果] Tool result too large, the result of this tool call call_7423df7024524a0b819234b5 was saved in the f...
# 本次处理的节点类型model
# [模型决策] 调用工具: read_file,参数为：{'file_path': '/large_tool_results/call_6586652948914a0aaaa1d8c3', 'offset': 0, 'limit': 200}
# [模型决策] 调用工具: read_file,参数为：{'file_path': '/large_tool_results/call_7423df7024524a0b819234b5', 'offset': 0, 'limit': 200}
# 本次处理的节点类型TodoListMiddleware.after_model
# 本次处理的节点类型tools
# [执行结果]      1	{"query": "人工智能 最新热点新闻 2025", "follow_up_questions": null, "answer": null, "images": [], "res...
# 本次处理的节点类型tools
# [执行结果]      1	{"query": "机器人 最新热点新闻 2025", "follow_up_questions": null, "answer": null, "images": [], "resu...
# 本次处理的节点类型model
# 📝 [最终回复] # 🤖 人工智能与机器人热点新闻汇总
#
# ---
#
# ## 一、人形机器人：从实验室走向产线
#
# ### 1. Genesis AI 发布首款通用型人形机器人 **Eno**
# Genesis AI 发布了其首款通用型人形机器人 **Eno**，被描述为"突破传统限制的下一代机器人"，标志着人形机器人从单一功能向通用能力的跨越。
#
# ### 2. Nvidia 发布开放人形机器人研究平台
# Nvidia 宣布推出 **Isaac GR00T Reference Humanoid Robot**，一个开放的人形机器人平台，旨在加速物理AI和通用机器人的研发。**Sharpa** 的 Wave 触觉灵巧手已集成到 **Unitree H2 Plus** 人形机器人参考设计中，成为首个基于 Nvidia 平台构建的灵巧人形平台。
#
# ### 3. 银河通用 Galbot S1 入驻宁德时代产线
# 宁德时代与银河通用签署全球战略合作协议，其重型人形机器人 **Galbot S1** 正式入驻宁德时代智能产线，具备双臂50公斤负载、纯视觉厘米级定位、360°全向避障能力，搭载宁德时代电池实现 **8小时超长续航**，已在量产线上 7×24 小时运行超过三个月。
#
# ### 4. X Square Robot：从翻跟头到叠衣服
# 中国企业 X Square Robot 正从展示性动作（后空翻、障碍跑）转向更具挑战性的日常任务（叠衣服），致力于构建具身智能缺失的"大脑"。该公司估值已超 **200亿元**。
#
# ---
#
# ## 二、具身智能与机器人融资热潮
#
# ### 5. AGILINK 五个月四轮融至独角兽
# 灵巧手公司 **AGILINK** 完成新一轮融资，投后估值突破 **10亿美元**。从成立到独角兽仅用五个月、四轮融资，创下灵巧手领域最快纪录。其 OmniHand 系列灵巧手累计交付超 **8,000台**，2026年Q1国内灵巧手市场出货量第一。
#
# ### 6. Standard Bots 估值达10亿美元
# 美国AI原生工业机器人制造商 **Standard Bots** 完成 **2亿美元** C轮融资，估值达到10亿美元。
#
# ### 7. 智元机器人（AI² Robotics）获近50亿元新融资
# 智元机器人完成近 **50亿元** 新一轮融资，持续加码具身智能赛道。
#
# ### 8. Seer Robotics（思灵机器人）港交所上市
# **思灵机器人**（06106.HK）在港交所主板上市，成为市场首个"机器人大脑"股票。上市首日涨超 **30%**，IPO募资约 **10.67亿港元**。其机器人控制器全球销量连续三年第一，全球市场份额达 **25%**。
#
# ---
#
# ## 三、AI 安全与国家安全
#
# ### 9. 五眼联盟警告：AI 正急剧改变网络威胁格局
# 2026年6月22日，五眼联盟（美、英、加、澳、新）发布联合警告，指出 AI 正在**将网络攻击周期从数年压缩到数月**。美国政府已对 Anthropic 的 Fable 5 和 Mythos 5 前沿AI模型实施出口限制，因其高级漏洞识别能力可能被滥用于加速网络攻击。
#
# ### 10. RLWRLD 入选世界经济论坛技术先锋
# 物理AI公司 **RLWRLD** 被评为 **2026年世界经济论坛技术先锋**，其自研的机器人基础模型 RLDX-1 正在推动物理AI基础设施建设。
#
# ---
#
# ## 四、AI 赋能制造业与工业
#
# ### 11. Nvidia 推出 AI 工厂管理蓝图
# Nvidia 发布全新软件蓝图，为制造商提供集中式 AI 系统，可实时监控、协调和优化全工厂运营。
#
# ### 12. Festo 推出 GripperAI 软件
# **Festo** 推出 AI 驱动的 **GripperAI** 软件，使机器人无需大量编程即可处理混合、陌生和随机摆放的产品。
#
# ### 13. Decart 发布 Oasis 3 世界模型
# 前沿AI实验室 **Decart** 发布最新世界模型 **Oasis 3**，将逼真的视觉环境引入机器人训练，弥合合成仿真与物理AI之间的差距。
#
# ### 14. CVPR 2026 收到超16,000篇论文投稿
# 全球顶级AI与计算机视觉会议 **CVPR 2026** 收到超过 **16,000篇** 论文投稿，反映AI技术研究的爆发式增长。
#
# ---
#
# ## 五、自动驾驶与无人机
#
# ### 15. Einride 自动卡车公司登陆纳斯达克
# 瑞典自动驾驶电动卡车公司 **Einride** 正式在纳斯达克上市。
#
# ### 16. Wing 与沃尔玛扩大无人机配送网络
# **Wing** 和 **沃尔玛** 将美国最大住宅无人机配送网络扩展至 **7个新都市区**。
#
# ### 17. QCraft 城市 NOA 系统在高通平台上量产
# **QCraft** 在高通 Snapdragon Ride 平台上成功演示城市自动导航辅助驾驶（NOA），目标2026年全球量产。
#
# ---
#
# ## 六、产业合作与生态建设
#
# ### 18. 京东与魔法原子战略合作
# **京东**与**魔法原子**签署战略合作协议，目标在京东平台实现 **10亿元** 品牌产品销售，共同推动消费级具身智能产品的大规模商业部署。
#
# ### 19. 宇树科技在苏州成立江苏首家具身智能产业学院
# **宇树科技**联合多方在苏州成立江苏省首家具身智能产业学院，配备 Go2 四足机器人、G1 人形机器人等高端设备，培养具身智能技术人才。
#
# ### 20. Geekplus 在丰田部署超400台 AMR
# **极智嘉（Geekplus）** 在日本多家丰田工厂部署了 **436台** 自主移动机器人（AMR），推动汽车制造业的智能化升级。
#
# ---
#
# ## 📊 趋势总结
#
# | 趋势方向 | 关键信号 |
# |---------|---------|
# | **人形机器人商业化** | Galbot S1 入驻宁德时代、Genesis AI 发布 Eno、Nvidia 开放平台 |
# | **具身智能资本狂热** | AGILINK 五个月成独角兽、智元50亿融资、思灵港交所上市 |
# | **AI 安全成国家安全议题** | 五眼联盟联合警告、前沿模型出口管制 |
# | **AI+制造深度融合** | Nvidia 工厂管理蓝图、Festo GripperAI、物理AI基础设施 |
# | **自动驾驶/无人机加速落地** | Einride 上市、Wing+沃尔玛扩张、QCraft 量产 |
#
# 当前AI与机器人领域呈现**技术突破、资本涌入、商业化加速**三重叠加态势，人形机器人和具身智能成为最热赛道，AI安全问题也上升到了国家安全层面。
# 本次处理的节点类型TodoListMiddleware.after_model