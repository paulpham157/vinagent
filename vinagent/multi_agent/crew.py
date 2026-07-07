from typing import Union, List, Awaitable, Any
import copy
from abc import ABC, abstractmethod
import logging
from vinagent.logger.logger import logging_message, logging_user_input
from vinagent.graph.operator import FlowStateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.utils.runnable import coerce_to_runnable
from langchain_together import ChatTogether
from langchain_core.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from vinagent.memory.history import InConversationHistory
from vinagent.oauth2.client import AuthenCard
from vinagent.multi_agent.base import CrewBaseAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrewAgent(CrewBaseAgent):
    """
    The CrewAgent class is designed to manage interactions with a language model, handle authentication, and process queries through a stateful graph-based workflow. It supports both synchronous and asynchronous query invocation and streaming responses.
    """

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        checkpoint: MemorySaver = None,
        graph: FlowStateGraph = None,
        flow: list[str] = [],
        num_buffered_messages: int = 10,
        authen_card: AuthenCard = None,
    ):
        """
        Args:
            llm (Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI]): The language model instance used for processing queries.
            checkpoint (MemorySaver, optional (default: None)): A memory checkpoint for saving and retrieving state information.
            graph (FlowStateGraph, optional (default: None)): A graph structure defining the workflow for processing queries.
            flow (list[str], optional (default: [])): A list of strings representing the flow of operations in the graph.
            num_buffered_messages (int): An buffered memory, which is not stored to memory, just existed in a runtime conversation. Default is a list of last 10 messages.
            authen_card (AuthenCard, optional (default: None)): An authentication card for verifying access tokens.
        """
        self.llm = llm
        self.checkpoint = checkpoint
        self.graph = graph
        self.flow = flow
        self.config = {
            "configurable": {"user_id": "unknown_user"},
            "thread_id": 123,
        }
        self.graph._apply_runtime_config(self.config)
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpoint, flow=self.flow
        )
        self.authen_card = authen_card
        self.in_conversation_history = InConversationHistory(
            messages=[], max_length=num_buffered_messages
        )

    def invoke(self, query: str, user_id: str, thread_id: str, **kwargs) -> dict:
        """
        Synchronously processes a query through the compiled graph after authentication.
        Args:
            query (str): The user query to process.
            user_id (str): The unique identifier for the user.
            thread_id (str): The thread identifier for the conversation.
            kwargs: Additional arguments passed to initialize_state.

        Returns:
            dict: The result of the graph invocation.
        """
        self.authenticate()
        input_state = self.initialize_state(query, user_id, thread_id)
        config = input_state.get("config", {})
        if config:
            self._compile_graph_with_config(config)
        return self.compiled_graph.invoke(**input_state)

    async def ainvoke(self, query: str, user_id: str, thread_id: str, **kwargs) -> dict:
        """Asynchronously processes a query through the compiled graph after authentication.
        Args:
            query (str): The user query to process.
            user_id (str): The unique identifier for the user.
            thread_id (str): The thread identifier for the conversation.
            kwargs: Additional arguments passed to initialize_state.

        Returns:
            dict: The result of the graph invocation.
        """
        self.authenticate()
        input_state = self.initialize_state(query, user_id, thread_id)
        config = input_state.get("config", {})
        if config:
            self._compile_graph_with_config(config)
        result = await self.compiled_graph.ainvoke(**input_state)
        return result

    def stream(self, query: str, user_id: str, thread_id: str, **kwargs) -> dict:
        """Streams the query processing results from the compiled graph after authentication.
        Args:
            query (str): The user query to process.
            user_id (str): The unique identifier for the user.
            thread_id (str): The thread identifier for the conversation.
            kwargs: Additional arguments passed to initialize_state.

        Returns:
            dict: The result of the graph invocation.
        """
        self.authenticate()
        input_state = self.initialize_state(query, user_id, thread_id)
        config = input_state.get("config", {})
        if config:
            self._compile_graph_with_config(config)
        result = []
        for chunk in self.compiled_graph.stream(**input_state):
            for v in chunk.values():
                if v:
                    result += v["messages"]
                    yield v
        return result

    def _compile_graph_with_config(self, config: dict):
        if config == self.config:
            return
        self.graph._apply_runtime_config(config)
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpoint, flow=self.flow
        )
        self.config = copy.deepcopy(config)
