# Asynchronous Invoke

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent-docs/blob/main/docs/tutorials/get_started/async_invoking.ipynb)

## Prerequisites


```python
%pip install vinagent
```

## Initialize LLM and Agent

To use a list of default tools inside [vinagent.tools](https://github.com/datascienceworld-kan/vinagent/tree/main/vinagent/tools) you should set environment varibles inside `.env` including `TOGETHER_API_KEY` to use llm models at togetherai site and `TAVILY_API_KEY` to use tavily websearch tool at tavily site:


```python
%%writefile .env
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```


```python
from vinagent.agent.agent import Agent
from langchain_together import ChatTogether
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))

# Step 1: Initialize LLM
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 2: Initialize Agent
agent = Agent(
    description="You are a Weather Analyst",
    llm = llm,
    skills = [
        "Update weather at anywhere",
        "Forecast weather in the futher",
        "Recommend picnic based on weather"
    ],
    tools=['vinagent.tools.websearch_tools'],
    tools_path = 'templates/tools.json', # Place to save tools. Default is 'templates/tools.json'
    is_reset_tools = True # If True, it will reset tools every time reinitializing an agent. Default is False
)
```

    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered search_api:
    {'tool_name': 'search_api', 'arguments': {'query': {'type': 'Union[str, dict[str, str]]', 'value': '{}'}}, 'return': 'Any', 'docstring': 'Search for an answer from a query string\n    Args:\n        query (dict[str, str]):  The input query to search\n    Returns:\n        The answer from search query', 'dependencies': ['os', 'dotenv', 'tavily', 'dataclasses', 'typing'], 'module_path': 'vinagent.tools.websearch_tools', 'tool_type': 'module', 'tool_call_id': 'tool_d697f931-5c00-44cf-b2f1-f70f91cc2973'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.websearch_tools


## Syntax for Async Invoke

Vinagent supports both synchronous (`agent.invoke`) and asynchronous (`agent.ainvoke`) execution methods. Synchronous calls block the main thread until a response is received, whereas asynchronous calls allow the program to continue running while waiting for a response. This makes asynchronous execution especially effective for I/O-bound tasks, such as when interacting with external services like search engine, database connection, weather API, .... In real-world usage, asynchronous calls can perform up to twice as fast as their synchronous counterparts.


```python
message = await agent.ainvoke("What is the weather in New York today?")
print(message.content)
```

## Latency Benchmarking

This is a performance benchmarking table based on 100 requests to [meta-llama/Llama-3.3-70B-Instruct-Turbo-Free](https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free) on TogetherAI. It demonstrates that the latency of `ainvoke` is nearly twice as fast as `invoke`. You may get different results due to the randomness of the requests and state of LLM-provider server.

| Number of requests | `ainvoke` (sec/req) | `invoke` (sec/req) |
|--------------------|---------------------|----------------------|
| 100                | 8.05-11.72          | 15.03-18.47          |

This is code for benchmarking between two inference methods. To save cost, we only run 5 times.


```python
import timeit
import asyncio

async def benchmark_ainvoke():
    message = await agent.ainvoke("What is the weather in New York today?")
    print(message.content)
    return message

def sync_wrapper():
    asyncio.run(benchmark_ainvoke())
    

execution_time = timeit.timeit(sync_wrapper, number=5)
print(f"Average execution of asynchronous time over 5 runs: {execution_time / 5:.2f} seconds")
```
    Average execution of asynchronous time over 5 runs: 8.93 seconds



```python
import timeit

def benchmark_invoke():
    message = agent.invoke("What is the weather in New York today?")
    print(message.content)

execution_time = timeit.timeit(benchmark_invoke, number=5)
print(f"Average execution of synchronous time over 5 runs: {execution_time / 5:.2f} seconds")
```
    Average execution of synchronous time over 5 runs: 15.47 seconds
