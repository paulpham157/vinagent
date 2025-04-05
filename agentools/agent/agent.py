import os
import re
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, Optional
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
import json
import importlib
import logging
from pathlib import Path
from typing import Union
from agentools.register.tool import ToolManager, register_function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """Abstract base class for agents"""
    
    @abstractmethod
    def __init__(self, llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI], tools: List = [], *args, **kwargs):
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
    
    TOOLS_PATH = Path("tool_template/tools.json")
    
    def __init__(self, llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI], tools: List = [], *args, **kwargs):
        """Initialize agent with LLM and tools list"""
        self.llm = llm
        self.tools = tools
        # Ensure tools file exists
        self.TOOLS_PATH.parent.mkdir(exist_ok=True)
        if not self.TOOLS_PATH.exists():
            self.TOOLS_PATH.write_text(json.dumps({}, indent=4), encoding='utf-8')
        self.register_tools(self.tools)
    
    def register_tools(self, tools: List[str]) -> Any:
        """
        Register a list of tools
        """
        for tool in tools:
            register_function(tool)


    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Select and execute a tool based on the task description"""
        try:
            tools = json.loads(self.TOOLS_PATH.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            tools = {}
            self.TOOLS_PATH.write_text(json.dumps({}, indent=4), encoding='utf-8')

        prompt = (
            "You are a function-calling AI agent. Select a function and its arguments "
            f"to solve this task based on available tools.\n"
            f"- Task: {query}\n"
            f"- Available tools: {json.dumps(tools)}\n"
            "- Only return dictionary without explaination and do not need bounded in ```python``` or ```json```"
            "{"
                '"tool_name": "The function",'
                '"arguments": "A dictionary of keyword-arguments to execute tool_name",'
                '"module_path": "module_path to import this tool"'
            "}"
        )

        try:
            response = self.llm.invoke(prompt).content
            tool_data = self._extract_json(response)
            
            if not tool_data or "None" in tool_data:
                return self.llm.invoke(query).content
                
            tool_call = json.loads(tool_data)
            return self._execute_tool(
                tool_call["tool_name"],
                tool_call["arguments"],
                tool_call["module_path"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None
        
    async def invoke_async(self, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's LLM"""
        return await self.llm.ainvoke(*args, **kwargs)

    def _execute_tool(self, tool_name: str, arguments: dict, module_path: str) -> Any:
        """Execute the specified tool with given arguments"""
        if module_path == '__runtime__' and tool_name in ToolManager._registered_functions:
            func = ToolManager._registered_functions[tool_name]
            return func(**arguments)
        
        try:
            if tool_name in globals():
                return globals()[tool_name](**arguments)
                
            module = importlib.import_module(module_path, package=__package__)
            func = getattr(module, tool_name)
            return func(**arguments)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract first valid JSON object from text using stack-based parsing"""
        start = text.find('{')
        if start == -1:
            return None
            
        stack = []
        for i in range(start, len(text)):
            if text[i] == '{':
                stack.append('{')
            elif text[i] == '}':
                stack.pop()
                if not stack:
                    return text[start:i + 1]
        return None
