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
result = deep_agent.invoke({
    "messages":[
        {"role":"user","content":f"{prompt}"}
    ]
})
# result = deep_agent.invoke({"inout":"人工智能和机器人的热点新闻！！"})
print(result['messages'][-1].content)
# # 🤖 人工智能与机器人最新热点新闻
#
# ---
#
# ## 一、人工智能领域
#
# ### 1. Google DeepMind CEO 预测：AGI 将在 2030 年前实现
# Google DeepMind CEO **Demis Hassabis** 预测，通用人工智能（AGI）将在 2030 年前实现。他表示："这将成为一项巨大的变革性技术，实际上将开启一个新的人类时代。" 他还强调，在人类级 AI 时代，**创造力与品味**将是人类最核心的竞争力。OpenAI CEO Sam Altman 此前也表达了类似观点，称到 2030 年，AI 系统将在大多数主要领域超越人类水平。
#
# ### 2. 五眼联盟发出 AI 网络安全威胁紧急警告
# 2026 年 6 月 22 日，美国国家安全局（NSA）牵头，五眼联盟发布联合声明，警告称 ** capable 的 AI 网络攻击模型距离投入使用仅数月之遥，而非数年**。声明指出，AI 正在根本性地改变攻击者与防御者之间的力量平衡，可能将网络攻击的时间线从数年压缩到数月，对电力网络、水务系统、医院和交通网络等关键基础设施构成前所未有的威胁。
#
# ### 3. 中国 AI 公司 Z.AI（智谱）快速追赶美国前沿模型
# 据 CNBC 报道，中国 AI 公司智谱的 **GLM 5.2** 模型在关键的智能体（Agentic）基准测试上正在缩小与美国前沿模型的差距，且该模型**免费开源**，采用速度已超过 DeepSeek。这对企业级 AI 应用和垂直领域 AI 产生了重要影响。
#
# ### 4. AI 泡沫争论升温
# 风投机构开始警告 AI 泡沫风险，重新评估 AI 初创公司的估值。讨论焦点包括：当前 AI 繁荣是否构成泡沫、投资者是否过度高估了初创公司的估值和 ARR 增长，以及 AI 如何改变了公司的评估、融资和增长方式。值得注意的是，**Menlo Ventures 因投资 Anthropic 获利，募资 30 亿美元新基金**。
#
# ### 5. AI 自我改进与治理挑战
# 随着 AI 系统能力不断增强，关于**递归自我改进（AI 自我训练）**的讨论日益增多——未来模型是否能够帮助开发自己的继任者？治理、监控和人类监督能否跟上这一快速发展的技术？这成为当前 AI 安全领域的核心议题。
#
# ---
#
# ## 二、机器人领域
#
# ### 6. AGIBOT 第 15,000 台机器人下线——具身智能里程碑
# 2026 年 6 月 28 日，中国具身 AI 领军企业 **AGIBOT（智元机器人）** 宣布其第 15,000 台机器人正式下线。从第 10,000 台到第 15,000 台的跨越，标志着具身 AI 机器人正从产品验证和批量生产阶段迈向**大规模交付和实际部署**。
#
# ### 7. 供应商瞄准 5 万亿美元人形机器人市场
# 据 Automotive News 报道，汽车零部件供应商正积极布局**人形机器人市场**，尽管对价值捕获存在担忧，但该市场的潜在规模高达 **5 万亿美元**。与此同时，汽车行业正在与 AI 争夺存储芯片——并且目前处于劣势。
#
# ### 8. 智元机器人（Seer Robotics）港交所上市，首日暴涨 30%
# **智元机器人（06106.HK）** 于 6 月 24 日在港交所主板上市，成为市场首只"机器人大脑"股票。上市首日开盘即飙升超过 30%，显示出市场对机器人赛道的强烈热情。该公司核心产品为机器人控制器（"机器人大脑"），2023-2025 年连续三年蝉联全球机器人控制器销量第一，全球市场份额达 **25%**。
#
# ### 9. 京东创始人刘强东：机器人将取代 70 万快递员
# 京东创始人**刘强东**在 APEC CEO 论坛上表示，随着自动化技术持续发展，机器人"迟早"将取代京东的 **70 万名快递员**。这是全球最大电商和物流公司领导者之一对配送工作未来做出的最明确判断之一。
#
# ### 10. Agility Robotics 以 25 亿美元估值通过 SPAC 上市
# 人形机器人初创公司 **Agility Robotics** 宣布通过 SPAC 方式上市，估值达 **25 亿美元**，成为人形机器人领域的重要资本事件。
#
# ### 11. FieldAI 机器人初创公司营收突破 1 亿美元
# 机器人软件初创公司 **FieldAI** 宣布营收和客户合同总额超过 **1 亿美元**。其软件可在人形机器人、机器狗、无人机和工业漫游车等多种机器人上运行，帮助它们在不可预测的环境中安全自主运行。
#
# ### 12. MWC 上海 2026：中国移动 AI 时代开启
# GSMA 在 MWC 上海 2026 上宣布移动行业进入**"IQ 时代"**——移动 AI 将重塑整个移动产业。中国三大运营商（中国移动、中国联通、中国电信）已在**人形机器人、低空经济（无人机）和自动驾驶**等领域积极部署 AI 技术。
#
# ---
#
# ## 三、趋势总结
#
# | 趋势方向 | 关键信号 |
# |---------|---------|
# | **AGI 时间线加速** | 多家 AI 巨头预测 2030 年前实现 |
# | **AI 安全威胁升级** | 五眼联盟罕见发出紧急警告 |
# | **具身智能规模化** | AGIBOT 1.5 万台下线，产业链加速成熟 |
# | **人形机器人资本热潮** | 5 万亿美元市场预期，多家公司上市 |
# | **中国 AI/机器人崛起** | 智谱 GLM 5.2 追赶前沿，智元港股上市 |
# | **AI 取代人工加速** | 京东 70 万快递员面临机器人替代 |
# | **AI 治理紧迫性上升** | 自我改进、网络安全、泡沫风险多重挑战 |