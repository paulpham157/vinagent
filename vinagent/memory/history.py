from typing import List, Optional
from langchain_core.messages import BaseMessage
from collections import deque


class InConversationHistory:
    def __init__(self, messages: List[BaseMessage] = [], max_length: int = 10):
        self.max_length = max_length
        self.history = deque(iterable=messages, maxlen=max_length)

    def _assign_iteration_id(self, message: BaseMessage, iteration_id: str) -> None:
        if iteration_id is not None:
            message.additional_kwargs["iteration_id"] = iteration_id

    def add_message(
        self, message: BaseMessage, iteration_id: str = None, is_rearrange: bool = False
    ) -> None:
        self._assign_iteration_id(message, iteration_id)
        self.history.append(message)
        if is_rearrange:
            self.rearrange()

    def add_messages(
        self,
        messages: List[BaseMessage],
        iteration_id: str = None,
        is_rearrange: bool = False,
    ) -> None:
        for msg in messages:
            self._assign_iteration_id(msg, iteration_id)
        self.history.extend(messages)
        if is_rearrange:
            self.rearrange()

    def rearrange(self) -> None:
        """
        Rearrange the conversation history so that:
        ToolMessages are placed immediately after their matching AIMessage (by tool_call_id).
        SystemMessage, HumanMessage, and the relative order of AIMessages are NOT altered.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        # 1. Isolate all ToolMessages
        tool_messages = [m for m in list(self.history) if isinstance(m, ToolMessage)]

        # 2. Rebuild the history iteratively
        new_history = []
        used_tools = set()

        for msg in list(self.history):
            if isinstance(msg, ToolMessage):
                # Skip tool messages; they will be inserted after their AIMessage
                continue

            new_history.append(msg)

            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    for tm in tool_messages:
                        msg_iter = getattr(
                            msg,
                            "iteration_id",
                            msg.additional_kwargs.get("iteration_id"),
                        )
                        tm_iter = getattr(
                            tm, "iteration_id", tm.additional_kwargs.get("iteration_id")
                        )
                        if (
                            getattr(tm, "tool_call_id", None) == tc_id
                            and id(tm) not in used_tools
                            and msg_iter == tm_iter
                        ):
                            new_history.append(tm)
                            used_tools.add(id(tm))
                            break

        # Append any orphaned ToolMessages at the end (e.g. error messages without corresponding AIMessage)
        for tm in tool_messages:
            if id(tm) not in used_tools:
                new_history.append(tm)

        self.history = deque(iterable=new_history, maxlen=self.max_length)

    def pop_left(self) -> None:
        self.history.popleft()

    def pop(self) -> None:
        self.history.pop()

    def get_history(self, max_history: Optional[int] = None) -> List[BaseMessage]:
        len_history = len(self.history)
        if max_history:
            return list(self.history)[-min(max_history, len_history) :]
        else:
            return list(self.history)

    def delete_history(self) -> None:
        """Clear all messages from the conversation history."""
        self.history.clear()

    def delete_message(self, index: int) -> BaseMessage:
        """
        Delete a message at the given index and return it.
        Supports negative indexing (e.g. -1 for the last message).
        Raises IndexError if the index is out of range.
        """
        history_list = list(self.history)
        removed = history_list.pop(index)
        self.history = deque(iterable=history_list, maxlen=self.max_length)
        return removed
