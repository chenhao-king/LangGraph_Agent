#Chatbot
# Objectives:
# use different message types -HumanMessage and AIMessage 
# Maintain a full conversation history using both message types (维护完整的历史对话记录)
# Use  @@ model using langchian's chatopenai
# create a sophisticated loop （构建一个复杂的循环）
# main goal :create a form of memory for our Agent （创建一种记忆形式）

import os
from typing import TypedDict,List,Union
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]   #是特殊的数据类型；既可以存储human 也可以存储aimessages
#   messages_ai:List[AImessages] 这个太长了 我们可以使用Union 并集
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
def process(state:AgentState) ->AgentState:
    """this node will solve your request you input """
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) # just extract the important content/stuff
    print(f"\nAI:{response.content}")
    return state

#代码解析：
# 关于状态的加工与更新的理解：
# 从右到左：提取结果中的纯文字正文、贴上AImessage 这个标签+通过.append 硬塞进这里列表的最末尾；
# 结果就是——一定要跳出来看 加工前与加工后的 state的区别：添加了AI的回答！！

graph = StateGraph(AgentState)
graph.add_node("response",process)
graph.add_edge(START,"response")
graph.add_edge("response",END)
app = graph.compile()

## 第二部分 也即是这里的难点 工厂外部的循环：
# 这里我们来具体的解析一下：


conversation_history = []    #这里理解为我们提前准备了一个空的背包；
user_input = input("Enter:") # 首先 这个是准备阶段：这里我们输入 “你好”——

# 进入循环 ——第一轮对话：
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input)) #往背包了加东西：[人类：“你好”]
    result = app.invoke({"messages":conversation_history}) # 公文包丢进工厂——AI 看到了“你好”，并在包里塞入了它的回答。 就是说 工厂只认知 message 
    conversation_history = result["messages"] #工厂吐出来的 result 长这样：{"messages": [人类："你好", AI："你好！很高兴见到你！"]}

    user_input = input("Enter:")

# 两个极其反直觉的“脱裤子放屁u”的操作：
# 首先 第一个就是 "messages":conversation_history 
# 大模型本身（llm）确实是可以直接看列表的。但是，你现在使用的是 LangGraph（图架构）。
# LangGraph 是一个极其严格的现代化流水线工厂。还记得你在第 16 行定下的规矩吗？

# 最后为什么要拿 result 去替换原来的包？ result = app.invoke({"messages":conversation_history})
# 真相：为了“安全隔离” ！！额 不多解释了 ；
