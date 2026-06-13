from langchain.agents import create_agent

from langchain_ollama import ChatOllama
 
from langchain_community.tools import tool

@tool

def get_weather(city:str)->str:

    """Get the current weather for a given city."""

    # This is a placeholder implementation. In a real implementation, you would

    # call a weather API to get the actual weather data.

    return f"The current weather in {city} is sunny with a temperature of 25°C."
 
print(get_weather.description)

print(get_weather.args)

print(get_weather.name)
 
llm=ChatOllama(model="gpt-oss:120b-cloud")
 
agent = create_agent(

    model=llm,

    tools=[get_weather],  

    system_prompt="You are a helpful assistant that provides weather information." \

    " Use the get_weather tool to get the current weather for a given city when asked."

    "do not answer from your knowledge, use the tool to get the weather information.")  
 
#agent
 
response=agent.invoke({"messages":[{"role": "user", "content": "What is the weather in Pune?"}]})
 
print(response)
 