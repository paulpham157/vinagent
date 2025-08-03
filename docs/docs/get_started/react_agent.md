# ReAct Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/react_agent.ipynb)

In reality, many complex use cases require combining reasoning and acting advancements to enable language models to solve various reasoning and decision-making tasks. Language models are getting better at both reasoning and acting, but these two directions have largely remained separate.

ReAct enables language models to generate both verbal reasoning traces and textual actions in an interleaved manner. While actions lead to observational feedback from an external environment (referred to as “Env” in the figure below), reasoning traces do not affect the external environment. Instead, they influence the model’s internal state by reasoning over the context and updating it with useful information to support future reasoning and actions.

![](../images/react_diagram.png)

ReAct is particularly useful in many use cases, such as solving math, logic, coding, and writing problems. That is why Vinagent offers a default ReAct Agent to facilitate their initialization for handling complex tasks that require interleaved reasoning and acting. You can learn more about ReAct in the original paper: [ReAct: Synergizing Reasoning and Acting in Language Models, Yao et al., 2022](https://arxiv.org/abs/2210.03629).

## Prerequisites

Let's install `vinagent` package and write environment variables to `.env` file.


```python
%pip install vinagent
```

Set environment varibles inside `.env` including `TOGETHER_API_KEY` to use llm models at [togetherai](https://api.together.ai/signin) site and `TAVILY_API_KEY` to use tavily websearch tool at [tavily](https://app.tavily.com/home) site:


```python
%%writefile .env
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```

## Create an ReAct Agent


```python
from langchain_together import ChatTogether 
from vinagent.agent.prebuilt import ReactAgent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

)
```

To demonstrate the efficient of ReAct Agent, let's assume we need to compute two dog weights average is `Husky` and `Bulldog`.


```python
%%writefile vinagent/tools/average_dogs.py
def weight_of_bulldog():
    """The weight of a bulldog"""
    return 25

def weight_of_husky():
    """The weight of a husky"""
    return 20

def average_weight_of_two_dogs(weight1: float, weight2: float):
    """The average weight of two dogs"""
    return (weight1 + weight2) / 2
```

Let's initialize the ReAct Agent and pass in the tools and the description of the agent.

```python
agent = ReactAgent(
    decription="You are a helpful assistant.",
    skills=["Search on internet"],
    tools=[
        'vinagent.tools.average_dogs',
        'vinagent.tools.websearch_tools'
    ],
    num_buffered_messages=10,
    llm=llm
)
```

    INFO:vinagent.register.tool:Registered weight_of_bulldog:
    {'tool_name': 'weight_of_bulldog', 'arguments': {}, 'return': '25', 'docstring': 'The weight of a bulldog', 'dependencies': [], 'module_path': 'vinagent.tools.average_dogs', 'tool_type': 'module', 'tool_call_id': 'tool_154b5559-0e7e-436b-9f9e-741df44e8416'}
    INFO:vinagent.register.tool:Registered weight_of_husky:
    {'tool_name': 'weight_of_husky', 'arguments': {}, 'return': '20', 'docstring': 'The weight of a husky', 'dependencies': [], 'module_path': 'vinagent.tools.average_dogs', 'tool_type': 'module', 'tool_call_id': 'tool_b4804fad-30cd-4a62-a019-869c70bd86ab'}
    INFO:vinagent.register.tool:Registered average_weight_of_two_dogs:
    {'tool_name': 'average_weight_of_two_dogs', 'arguments': {'weight1': 0.0, 'weight2': 0.0}, 'return': 'None', 'docstring': 'The average weight of two dogs', 'dependencies': [], 'module_path': 'vinagent.tools.average_dogs', 'tool_type': 'module', 'tool_call_id': 'tool_b77b87ff-1fef-4ee1-8a62-ebbde8dcfc00'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.average_dogs
    INFO:vinagent.register.tool:Registered search_api:
    {'tool_name': 'search_api', 'arguments': {'query': '{}'}, 'return': 'Any', 'docstring': 'Search for an answer from a query string\n    Args:\n        query (dict[str, str]):  The input query to search\n    Returns:\n        The answer from search query', 'dependencies': ['os', 'dotenv', 'tavily', 'dataclasses', 'typing'], 'module_path': 'vinagent.tools.websearch_tools', 'tool_type': 'module', 'tool_call_id': 'tool_2ec9aaa3-8585-4aec-b5f2-f86335c1f99a'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.websearch_tools


Let's ask the average weight of two dogs.

```python
answer = agent.invoke(query="What is the average weight of two dogs Husky and Bulldog?")
answer
```

    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'weight_of_husky', 'tool_type': 'module', 'arguments': {}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool weight_of_husky({})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'weight_of_bulldog', 'tool_type': 'module', 'arguments': {}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool weight_of_bulldog({})
    INFO:vinagent.agent.agent:Tool calling iteration 3/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'average_weight_of_two_dogs', 'tool_type': 'module', 'arguments': {'weight1': 20.0, 'weight2': 25.0}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool average_weight_of_two_dogs({'weight1': 20.0, 'weight2': 25.0})
    INFO:vinagent.agent.agent:Tool calling iteration 4/10

    AIMessage(content='I now know the final answer\nFinal Answer: The average weight of a Husky and a Bulldog is 22.5.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 28, 'prompt_tokens': 3179, 'total_tokens': 3207, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--c2183de6-749b-4a3e-9331-74820095dc18-0', usage_metadata={'input_tokens': 3179, 'output_tokens': 28, 'total_tokens': 3207, 'input_token_details': {}, 'output_token_details': {}})


## Reasoning over history

The ReAct agent can save the user's chat history. This enables the agent to reason over the retained conversational context, which often contains important information in previous interactions. In this case, we set `num_buffered_messages=10`, meaning it can store up to 10 messages in the chat history. In the following example, we assume the Husky's weight has changed. The ReAct agent will reason over the chat history to determine the new average weight.


```python
answer = agent.invoke(query="If the average weight of a Husky is 30 kilograms, by how many kilograms does the average weight of the two dogs increase?")
answer
```

    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'weight_of_bulldog', 'tool_type': 'module', 'arguments': {}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool weight_of_bulldog({})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'average_weight_of_two_dogs', 'tool_type': 'module', 'arguments': {'weight1': 30.0, 'weight2': 25.0}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool average_weight_of_two_dogs({'weight1': 30.0, 'weight2': 25.0})
    INFO:vinagent.agent.agent:Tool calling iteration 3/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'average_weight_of_two_dogs', 'tool_type': 'module', 'arguments': {'weight1': 20.0, 'weight2': 25.0}, 'module_path': 'vinagent.tools.average_dogs'}
    INFO:vinagent.register.tool:Completed executing module tool average_weight_of_two_dogs({'weight1': 20.0, 'weight2': 25.0})
    INFO:vinagent.agent.agent:Tool calling iteration 4/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': '27.5 - 22.5'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': '27.5 - 22.5'})
    INFO:vinagent.agent.agent:Tool calling iteration 5/10
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 5 iterations.

    AIMessage(content='The increase in the average weight of the two dogs is 5 kilograms.\n\nThought: I now know the final answer\nFinal Answer: The average weight of the two dogs increases by 5 kilograms.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 41, 'prompt_tokens': 3469, 'total_tokens': 3510, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--4231bdd1-f0c4-492d-a716-af74a144b0ca-0', usage_metadata={'input_tokens': 3469, 'output_tokens': 41, 'total_tokens': 3510, 'input_token_details': {}, 'output_token_details': {}})



## Reasoning with search engine

Test with another use case that compares the populations of the three cities: `New York`, `Beijing`, and `Hanoi`.


```python
answer = agent.invoke(query="Which city has the greatest population among New York, Beijing, and Hanoi?")
answer
```

    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'population of New York'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'population of New York'})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'population of Beijing'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'population of Beijing'})
    INFO:vinagent.agent.agent:Tool calling iteration 3/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'population of Hanoi'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'population of Hanoi'})
    INFO:vinagent.agent.agent:Tool calling iteration 4/10 
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 4 iterations.

    AIMessage(content='We have the populations of New York, Beijing, and Hanoi. New York has a population of approximately 8.5 million, Beijing has a population of around 21.8 million, and Hanoi has a population of around 8,807,523.\n\nThought: Comparing these numbers, we can see that Beijing has the greatest population among the three cities.\n\nThought: I now know the final answer\nFinal Answer: Beijing has the greatest population among the three cities, with a population of approximately 21.8 million.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 109, 'prompt_tokens': 3331, 'total_tokens': 3440, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--28a56b66-22bc-420e-8be8-914992b322fe-0', usage_metadata={'input_tokens': 3331, 'output_tokens': 109, 'total_tokens': 3440, 'input_token_details': {}, 'output_token_details': {}})



Test with a math problem that requires reasoning and acting to be interleaved across multiple cycles.


```python
answer = agent.invoke(query="Vinagent is a community project of DataScienceWorld.Kan. It is an initiative by the founder of DataScienceWorld.Kan. Who is the leader of Vinagent?")
answer
```

    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'DataScienceWorld.Kan founder'}, 'module_path': 'vinagent.tools.websearch_tools'}
    INFO:vinagent.register.tool:Completed executing module tool search_api({'query': 'DataScienceWorld.Kan founder'})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 2 iterations.

    AIMessage(content='Since Pham Đinh Khanh is the founder of DataScienceWorld.Kan and Vinagent is an initiative of this founder, we can conclude that Pham Đinh Khanh is the leader of Vinagent.\n\nThought: I now know the final answer\nFinal Answer: Pham Đinh Khanh is the leader of Vinagent.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 69, 'prompt_tokens': 2889, 'total_tokens': 2958, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--b9a36ebc-aa2f-42c5-a6d8-2a3633da41e9-0', usage_metadata={'input_tokens': 2889, 'output_tokens': 69, 'total_tokens': 2958, 'input_token_details': {}, 'output_token_details': {}})


