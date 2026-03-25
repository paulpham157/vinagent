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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrewBaseAgent(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        checkpoint: MemorySaver = None,
        graph: FlowStateGraph = None,
        flow: list[str] = [],
        authen_card: AuthenCard = None,
        *args,
        **kwargs,
    ):
        """Initialize a new Agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, user_id: str, thread_id: str = 123, **kwargs) -> Any:
        """Synchronously invoke the agent's main function"""
        pass

    @abstractmethod
    async def ainvoke(
        self, query: str, user_id: str, thread_id: str = 123, **kwargs
    ) -> Awaitable[Any]:
        """Asynchronously invoke the agent's main function"""
        pass

    @abstractmethod
    def stream(self, query: str, user_id: str, thread_id: str = 123, **kwargs) -> Any:
        """Streaming the agent's main function"""
        pass

    def authenticate(self):
        """
        Verifies access using the provided authen_card. If no card is provided, authentication is skipped.

        Returns
            bool: True if authentication succeeds or is skipped, otherwise raises an exception.
        Raises
            Exception: If authentication fails.

        Logs
            Info log if no authentication card is provided.
            Info log for successful or failed authentication.
        """
        if self.authen_card is None:
            logger.info("No authentication card provided, skipping authentication")
            return True

        is_enable_access = self.authen_card.verify_access_token()
        if is_enable_access:
            logger.info(f"Successfully authenticated!")
        else:
            logger.info(f"Authentication failed!")
            raise Exception("Authentication failed!")
        return is_enable_access

    def initialize_state(
        self, query: str, user_id: str, thread_id: str = 123, **kwargs
    ):
        """Prepares the input state and configuration for query processing.
        Args:
            query (str): The user query to process.
            user_id (str): The unique identifier for the user.
            thread_id (str, optional (default: "123")): The thread identifier for the conversation.

        Returns: A dictionary containing
            input: The input state for the graph.
            config: The configuration for the graph.
        """
        input_state = (
            kwargs["input_state"]
            if "input_state" in kwargs
            else {"messages": {"role": "user", "content": query}}
        )

        config = (
            kwargs["config"]
            if "config" in kwargs
            else {
                "configurable": {"user_id": user_id},
                "thread_id": thread_id,
            }
        )

        return {"input": input_state, "config": config}
