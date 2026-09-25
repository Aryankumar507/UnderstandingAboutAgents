from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient
import os
import requests

load_dotenv()

#Import Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("tavily api key not found")

if not WEATHER_API_KEY:
    raise ValueError("weather key not found")
#Makign Tools
# 1. Weather tool
@tool
def get_weather(city: str) -> str:
    """Get current weather of the city"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    response = requests.get(
        url,
        params={"q": f"{city},IN", "appid": WEATHER_API_KEY, "units": "metric"}
    )

    if response.status_code != 200:
        raise ValueError("API key could not fetch the data")

    data = response.json()

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"

# 2. Tavily news tool 
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def get_news(city:str)->str:
    """Get the latest news about the city"""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    result = response.get("results",[])
    if not result:
        return f"No news found for city {city}"
    newlist = []
    for r in result:
        title = r.get("title","NO title")
        url = r.get("url","")
        snippet = r.get("content","")

        newlist.append(f"{title}\n{url}\n{snippet}")
    return f"latest news in city {city}:\n\n" + "\n\n".join(newlist)

# tools = {
#     "get_weather":get_weather,
#     "get_news":get_news
# }

# def access_tool() -> dict:
#     return tools