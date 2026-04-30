import os #用来读取环境变量；
from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv #used to store secret stuff like API keys or configuration values.
# we need a api to communicate with the llm in their could servers;

#加载 .env 文件中的密钥；
load_dotenv()

class AgentState(TypedDict):
    messages:List[HumanMessage]

llm = ChatOpenAI(
    model="deepseek-chat", # 指定deepseek 的模型
    api_key=os.getenv("DEEPSEEK_API_KEY"), #从 .env 中拿钥匙
    base_url="https://api.deepseek.com" #指向Deepseek 的服务器大门
)
# 定义处理车间
def process(state:AgentState) ->AgentState:
    """this node that process the message and llm'response """
    response = llm.invoke(state['messages'])
    print(f"\nAI:{response.content}")
    return state
# 代码解析：
# 1）node function 与之前设立的区别：
# 2）我们对于这个node function 的理解一定是：拆包、加工、重新包装；
# 3）llm.invoke——调用大模型———赋值给 response 
# 4）response.content—— 对象属性访问——大模型返回的是一个高级的object ——有各种属性的标签 这里我们只取出 content 正文部分；
# 5）\n 特殊的转义字符语法 表示 换行 
 

#构建图纸
graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter:") #input 是python 的一个内置函数、必须带括号、括号里面放一段引号抱起来的提示词、只要代码执行这个工具、程序就会立刻暂停、允许用户在键盘上打字；
while user_input != "exit":  #while 条件循环—— ！= 不输入 /不等于 这个exit：
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")

# 区别·一下  赛包——invoke（）
# 之前：app.invoke({"player_name": "Student", "attempts": 0, "lower_bound": 1})
# 现在：
# agent.invoke({"messages": [HumanMessage(content=user_input)]})
# 都是括号里面 使用python字典使用大括号 {} 包起来的键对值；
# 只不过这个values 不同 ——或者理解为处理得对象不同：
# 拆解 [HumanMessage(content=user_input)]
# 首先是已经定义好得：messages 这个变量 ——其次是
# 因为是 List 列表 所以有这个[]。其次 HumanMessage ——在这个LangChain这个体系——有三种身份标签：
# HumanMessage AIMessage SystemMessage
# 所以这个：[HumanMessage(content=user_input)] 