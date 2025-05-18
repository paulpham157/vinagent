import json
import importlib
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, Optional
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
import logging
from pathlib import Path
from typing import Union
from vinagent.register.tool import ToolManager
from vinagent.memory.memory import Memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List = [],
        *args,
        **kwargs,
    ):
        """Initialize a new Agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Synchronously invoke the agent's main function"""
        pass

    @abstractmethod
    async def invoke_async(self, query: str, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's main function"""
        pass


class Agent(AgentMeta):
    """Concrete implementation of an AI agent with tool-calling capabilities"""
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List = [],
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools = False,
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        skills: list[str] = ["You can answer the user question with tools"],
        memory_path: Path = None,
        is_reset_memory = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the agent with a language model, a list of tools, a description, and a set of skills.
        Parameters:
        ----------
        llm : Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI]
            An instance of a language model used by the agent to process and generate responses.

        tools : List, optional
            A list of tools that the agent can utilize when performing tasks. Defaults to an empty list.

        tools_path: Path, optional
            The path to the file containing the tools. Defaults to a template file.

        description : str, optional
            A brief description of the assistant's capabilities. Defaults to a general helpful assistant message.

        skills : list[str], optional
            A list of skills or abilities describing what the assistant can do. Defaults to a basic tool-usage skill.

        is_reset_tools : bool, optional
            A flag indicating whether the agent should override its existing tools with the provided list of tools. Defaults to False.

        memory_path : Path, optional
            The path to the file containing the memory. Defaults to a template file. Only valid if memory is not None.

        is_reset_memory : bool, optional
            A flag indicating whether the agent should reset its memory when re-initializes it's memory. Defaults to False. Only valid if memory is not None.

        *args, **kwargs : Any
            Additional arguments passed to the superclass or future extensions.
        """

        self.llm = llm
        self.tools = tools
        self.description = description
        self.skills = skills
        self.tools_path = None
        if tools_path:
            self.tools_path = Path(tools_path) if isinstance(tools_path, str) else tools_path
        else:
            self.tools_path = Path("templates/tools.json")
        
        self.is_reset_tools = is_reset_tools
        self.tools_manager = ToolManager(tools_path=self.tools_path, is_reset_tools=self.is_reset_tools)

        self.register_tools(self.tools)
        if memory_path and (not memory_path.endswith(".json")):
            raise ValueError("memory_path must be json format ending with .json. For example, 'templates/memory.json'")
        self.memory_path = Path(memory_path) if isinstance(memory_path, str) else memory_path
        self.is_reset_memory = is_reset_memory
        self.memory = None
        if self.memory_path:
            self.memory =Memory(
                memory_path=self.memory_path,
                is_reset_memory=self.is_reset_memory
            )
        self._user_id = None
        
    def register_tools(self, tools: List[str]) -> Any:
        """
        Register a list of tools
        """
        for tool in tools:
            self.tools_manager.register_tool(tool)

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, new_user_id):
        self._user_id = new_user_id

    def invoke(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", *args, **kwargs) -> Any:
        """
        Select and execute a tool based on the task description
        """
        if self._user_id:
            pass
        elif user_id == "unknown_user": # User forgot input their name
            self._user_id=input("You forgot clarifying your name. Input user_id:")
        else: # user clarify their name
            self._user_id=user_id
        logger.info(f"I'am chatting with {self._user_id}")

        try:
            tools = json.loads(self.tools_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
        
        prompt = (
            "You are given a task, a list of available tools, and the memory about user to have precise information.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(tools)}\n"
            f"- Memory: \n{self.memory.load_memory(load_type='string', user_id=self._user_id) if self.memory else ''}\n"
            f"- User: {self._user_id}\n"
            "------------------------\n"    
            "Instructions:\n"
            "- Let's answer in a natural, clear, and detailed way without providing reasoning or explanation."
            "- If user used I in Memory, let's replace by name {self._user_id} in User part."
            "- You need to think about whether the question need to use Tools?"
            "- If it was daily normal conversation. Let's directly answer as a human with memory.\n"
            "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "Let's say I don't know and suggest where to search if you are unsure the answer.\n"
            "Not make up anything.\n"
        )
        
        skills = "- ".join(self.skills)
        messages = [
            SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
            HumanMessage(content=prompt),
        ]

        if self.memory and is_save_memory:
            self.memory.save_short_term_memory(self.llm, query, user_id=self._user_id)

        try:
            response = self.llm.invoke(messages)
            tool_data = self.tools_manager.extract_json(response.content)

            if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                return response

            tool_call = json.loads(tool_data)
            return self._execute_tool(
                tool_call["tool_name"], tool_call["arguments"], tool_call["module_path"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

    async def invoke_async(self, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's LLM"""
        return await self.llm.ainvoke(*args, **kwargs)

    def _execute_tool(self, tool_name: str, arguments: dict, module_path: str) -> Any:
        """Execute the specified tool with given arguments"""
        # If function is directly registered by decorator @function_tool. Access it on runtime context.
        registered_functions = self.tools_manager.load_tools()

        if (
            module_path == "__runtime__"
            and tool_name in self.tools_manager._registered_functions
        ):
            func = self.tools_manager._registered_functions[tool_name]
            content = f"Completed executing tool {tool_name}({arguments})"
            logger.info(content)
            artifact = func(**arguments)
            tool_call_id = registered_functions[tool_name]["tool_call_id"]
            message = ToolMessage(
                content=content, artifact=artifact, tool_call_id=tool_call_id
            )
            return message

        # If function is imported from a module, access it on module path.
        # try:
        if tool_name in globals():
            return globals()[tool_name](**arguments)

        module = importlib.import_module(module_path, package=__package__)
        func = getattr(module, tool_name)
        artifact = func(**arguments)
        content = f"Completed executing tool {tool_name}({arguments})"
        logger.info(content)
        tool_call_id = registered_functions[tool_name]["tool_call_id"]
        message = ToolMessage(
            content=content, artifact=artifact, tool_call_id=tool_call_id
        )
        # if self.memory and isinstance(message.artifact, str):
        #     logging.info(message.artifact)
        #     self.memory.save_short_term_memory(self.llm, message.artifact)
        # else:
        #     logging.info(message.artifact)
        #     self.memory.save_short_term_memory(self.llm, message.content)
        return message
        # except (ImportError, AttributeError) as e:
        #     logger.error(f"Error executing tool {tool_name}: {str(e)}")
        #     return None

    def function_tool(self, func: Any):
        return self.tools_manager.function_tool(func)
