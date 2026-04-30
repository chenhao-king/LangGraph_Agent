#   ReAct Agent   stands for reasoning and acting 反应代表推理和行动；
#   Learn how to create tools in langgraph (怎样去创建工具)
#   How to create a react graph (创建一个推理加行动的agent)
#   work with different tpyes of messages such as toolmessages Humanmessage AImessages
#   konw the robustness and creast a robust ReAct Agent



import os
from typing import Annotated,Sequence,TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

#代码解析
# 1）各种Message ——结合我们之前学的 HumanMessage 和AIMessage 这里又添加了
# SystemMessage -系统消息 /人设剧本-用户看不见 是内部给llm设定的“人设”
# ToolMessage -工具的汇报单 理解为 工具执行完毕后的汇报回执单；
# BaseMessage -所有消息的“祖宗”、是一个抽象的底层基类：例如：之前写的List[Union[hunammessage,AImessage]]
# 这里就可以直接写成 Sequence[BaseMessage] 意思就是 只要是消息家族的，统统可以装进这个队伍里
# 2）add_messages 与生态管理
# 3）Annotated -语法含义是带有注释/规则的数据类型；
# 4）@tool 装饰器—— 可以将一个普通的死板的代码快——贴上@tool 这个装饰器 ——变成一个llm可以读懂的、而且可以自主调用的“AI专属工具”
# 5）ToolNode 工具代工厂：之前我们将node 写进graph 里面需要手动写一个 def precession（）：函数——
# 



load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

# 代码解析：
# Sequence(序列)：可以理解为高级版的List（列表队伍）
# add_messages :自动缝合函数——可以理解为是 .append()的升级版—— 只有有了新的消息、就会自动识别并缝合到现有的对话记录后面；
# decorator 装饰器 - this function is quite special:ti's a tool which we're going to use
# “我声明一个 messages 的变量，它是一个装满各路消息的队伍（Sequence）。并且，我给它绑定一个契约：
# 以后不管谁往这个队伍里扔新消息，都必须交给 add_messages 这个自动缝合机来处理（Annotated）。”
# ！！ 深度理解：如果没有这里的 Annotated langgraph 的潜规则是覆盖机制（Overwrite）导致你的llm自有最新的一句话记忆；
#       为了打破这个默认的潜规则 ——这里需要使用annotated 贴附加说明的标签；
#       警告！不要直接的覆盖——而是使用这个 add_messages 把新内容缝合到旧内容的后面；



@tool
def add(a:int,b:int):
    """this is an addition function that adds 2 numbers together"""

    return a + b
tools = [add]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

model = llm.bind_tools(tools) 


def model_call(state:AgentState) ->AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant,place answer my query to the best of your ability"                                                              
    )  
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}
# response = model.invoke(["You are my AI assistant,place answer my query to the best of your ability"])

def should_continue(state:AgentState) ->str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)
graph.add_edge("tools","our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","Add 40 + 12 and then multiply the tresult by 6 also tell me a joke place")]}
print_stream(app.stream(inputs,stream_mode="values"))