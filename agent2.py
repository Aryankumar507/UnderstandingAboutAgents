from tools import get_news,get_weather
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("mistral api key not found")

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite",api_key = GOOGLE_API_KEY,thinking_budget=0,streaming = True)

@wrap_tool_call
def human_approval(request,handler):
    """Ask for Human approval before every tool call"""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call {tool_name}. Approve? (yes/no): ")
    if confirm.lower() == "no":
        return ToolMessage(
            content = "Tool call denied by user",
            tool_call_id = request.tool_call["id"]
        )
    return handler(request)


my_agent = create_agent(
    model = llm,
    tools = [get_news,get_weather],
    system_prompt="You are an helpful city assistant",
    middleware=[human_approval]
)



print("City Agent: Type exit to quit")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    result = my_agent.invoke({
        "messages":[{
            "role":"user",
            "content":user_input
        }]
    })
    last_message = result['messages'][-1]
    print(last_message.text)


