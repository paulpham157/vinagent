# Add tools
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent-docs/blob/main/docs/tutorials/get_started/add_tool.ipynb)

## Prerequisites

Install `vinagent` library

```python
%pip install vinagent
```

## Tool types

`Vinagent` allows you to connect to three types of tools:

- Function tool: A Python function is registered into a specific agent using a decorator.
- Module tool: A function from a Python module, saved in a specific folder, can be registered as a tool.
- MCP tool: Create an MCP tool, which connects to an MCP server using the MCP protocol.


## Example of module tool

You can add module tools from a Python module path as follows:
- Initialize an LLM model, which can be any model wrapped by the [Langchain BaseLLM](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html) class. I use [TogetherAI](https://api.together.ai/signin) chat model in there, thus, you need to create `.env` environment with variable
```
TOGETHER_API_KEY="Your together API key"
```
You can use other LLM Provider API as long as it was initialized by Langchain `BaseLLM` class.


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
```

- Initialize an Agent with tools, which are wrapped inside the tools argument as a list of paths:



```python
import os
os.makedirs('./tools', exist_ok=True)
```


```python
%%writefile tools/hello.py
def hello_from_vinagent():
    '''A greet of Vinagent to everyone'''
    return "Hello my cute cute friend, I'm vinagent and I am here to play with you 😄!"
```

    Writing tools/hello.py



```python
# Step 1: Create Agent with tools
agent = Agent(
    description="You are a Vinagent",
    llm = llm,
    skills = [
        "Friendly talk with anyone"
    ],
    tools = ['tools/hello.py'],
    tools_path = 'templates/tools.json',
    is_reset_tools = True
)
```

    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered hello_from_vinagent:
    {'tool_name': 'hello_from_vinagent', 'arguments': {}, 'return': "Hello my cute cute friend, I'm vinagent and I am here to play with you 😄!", 'docstring': 'A greet of Vinagent to everyone', 'dependencies': [], 'module_path': 'vinagent.tools.hello', 'tool_type': 'module', 'tool_call_id': 'tool_a25e45c3-81df-4b68-982d-d308c403a725'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.hello


!!! note
    `tools_path` is where the tools dictionary is saved. The default value is templates/tools.json.

!!! tip "Resetting Your Tools"
    If you set `is_reset_tools = True`, it will override the tool definitions every time an agent is reinitialized.


## Asking tool

```python
# Step 2: invoke the agent
message = agent.invoke("Hi Vinagent, Can you greet by your style?")
print(message.content)
```

    
    Hello my friend, I'm vinagent, an AI smart assistant. I come here to help you 😄!
