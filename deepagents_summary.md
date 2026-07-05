1、DeepAgent智能体创建
deep_agent = create_deep_agent(

    model="",
    system_prompt="",
    checkpoint = MemorySaver(),
    skill=[],
    tool=[],
    middleware=[],
    interrupt_on=｛｝,
    subagent=[],
    backend=None,
    store=None,

)

属性：

模型、系统提示词、记忆点、skill、tool、中间件、interrupt_on、subagent、backend长期记忆、store

2、执行和结果处理

同步执行：invoke | stream

异步执行：ainvoke | astream

结果处理：

invoke：result['message'][-1].content

stream: 根据key判断时工具调用还是模型调用 

    key=tools -> ToolMessage 调用工具
    key=model -> AIMessage 调用工具或者子智能体或者直接返回结果（直接返回结果content非空，
                            tool_calls非空调用工具(调用子智能体被包装成了工具，工具名为task)，）
                            content和tool_calls不会同时有值

3、创建子智能体，以及兼容LangChain和LangGraph

使用字典或者CompiledSubAgent对象包装graph或者chain

字典：直接创建一个{}对象，然后添加对应属性即可
![img.png](assert/img.png)

CompiledSubAgent：
![img_1.png](assert/img_1.png)
需要注意的是graph的state状态中必须包含一个叫做messages的属性

4、人机交互 HITL

(1)必须配置短期记忆 checkpointer = MemorySaver()  生产可使用redis

(2)必须配置线程id，用于中断回复

(3)创建agent的时候设置 interrupt_on 的字典参数，｛"工具名":True|Flase｝，True则交互，否则直接放行

(4)执行result = invoke()的时候，如果出发人机交互，这次不会真的执行工具

(5)获取模型返回的交互和拦截参数 
    
    result = result['__interrupt__'].[0].value -> 
        {action_request:[{name,args}，｛｝...] | review_config}
    
    decision = [{"type":"edit",edited_action:{name：工具名，args：｛参数名：新的参数｝}}]

(6)再次执行result = invoke(Command(resume={"decisions}:[]),config=上次线程id)


5、长短期记忆

短期记忆：一次会话执行的过程 使用checkpointer = MemorySaver()

长期记忆：执行的最终结果，使用store 和 backend 实现跨会话的数据共享

6、中间件

自定义中间件：@warp_tool_call 函数(request,handler) 前置增强 result=handler(request) 后置增强 return结果

官方写好的中间件：

https://reference.langchain.com/python/langchain/middleware

提供了很多常用的中间件实现，比如：

SummarizationMiddleware：可以对工具调用进行总结（指定token（上下文）的压缩阈值（通常为模型上下文窗口的三分之二）以及对话轮数阈值），帮忙提取上下文摘要，进行压缩

Model Call Limit Middleware：可以指定线程id限制工具在一次回话（一次invoke执行）或者一个线程调用模型的次数，可以指定exit_behavior参数为end-退出或者error-抛异常

Tool call limit middleware: 可以指定线程id调用工具一次回话（一次invoke执行）或者一个线程调用的次数、

HumanInTheLoopMiddleware：可以实现人类在回路中的中间件

使用：

deepagent的middleware的参数[中间件1,中间件2...]

7、skill和配置

skill就是提示词，本质上是一个sop，作用就是规范流程和提示词服用

文件夹名 = SKILL.md 的name

元数据要简洁明了

配置：

使用filesystembackend链接本地文件，然后使用skills参数指定skill所在文件夹

加载原理：

渐进式披露