from tools import get_news,get_weather
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,ToolMessage,SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("mistral api key not found")

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

tools = {
    "get_news":get_news,
    "get_weather":get_weather
}

# llm = ChatMistralAI(model = "mistral-small-latest",api_key=MISTRAL_API_KEY)
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite",api_key = GOOGLE_API_KEY,thinking_budget=0)
llm_with_tool = llm.bind_tools([get_weather,get_news])


def get_text(result):
    if isinstance(result.content, str):
        return result.content
    return "".join(
        block["text"] for block in result.content
        if isinstance(block, dict) and block.get("type") == "text"
    )



#Agent Loop 
messages = []
messages = [
    SystemMessage(content=(
    "You are a concise city information assistant. "
    "When reporting news, always include the article links exactly as given. "
    "When reporting weather, answer in a short natural sentence — never dump raw JSON."
))
]
print("City Intelligence System")
print("type exit to quit")

while True:
    user_input = input("user: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                #human in the loop 
                confirm = input(f"Agent wants to call {tool_name}. Approve(yes/no): ")
                if confirm.lower() == "no":
                    messages.append(ToolMessage(
                        content="Tool call denied by user.",
                        tool_call_id=tool_call["id"]
                    ))
                    print("tool call denied and I cannot get the latest information")
                    continue
                tool_result = tools[tool_name].invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(tool_result),tool_call_id = tool_call["id"]))
        else:
            print(get_text(result))
            break


