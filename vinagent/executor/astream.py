from typing import Union, AsyncGenerator
import asyncio
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from vinagent.executor.base import AgentResponse
from vinagent.executor.base import MessageHandler
from vinagent.executor.base import AsyncStreamInvokeExecutorBase
from vinagent.logger.logger import logger
from vinagent.register.tool import ToolCall
from vinagent.executor.guardrail import GuardrailExecutor
from vinagent.register.tool import ToolManager
from vinagent.message.adapter import adapter_ai_response_with_tool_calls
from vinagent.prompt.agent_prompt import PromptHandler, PromptToolResult
from vinagent.memory.history import InConversationHistory
from vinagent.memory.memory import Memory
from vinagent.mcp.client import DistributedMCPClient


class AsyncStreamInvokeExecutor(
    AsyncStreamInvokeExecutorBase, MessageHandler, PromptHandler
):
    """
    Async streaming variant of the agent executor.

    Provides async generator methods for Steps 1 and 3 so that tokens are
    yielded to the caller (e.g. a WebSocket handler) as soon as they arrive
    from the LLM.  Step 2 (tool execution) is purely async I/O and does not
    stream tokens, so it is identical to ``AsyncInvokeExecutor``.
    """

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        guardrail_executor: GuardrailExecutor = None,
        seconds_limit: int = 30,
        *args,
        **kwargs,
    ):
        self.llm = llm
        self.guardrail_executor = guardrail_executor
        self.seconds_limit = seconds_limit

    # ------------------------------------------------------------------
    # Step 1 — determine tool vs. direct answer (structured output, sync-safe)
    # ------------------------------------------------------------------

    async def define_tools_async(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        """Invoke the LLM with structured output to get an AgentResponse."""
        messages = self._sanitize_history(messages)
        structured_llm = self.llm.with_structured_output(
            AgentResponse, method="function_calling"
        )
        try:
            response = await structured_llm.ainvoke(messages)
            if isinstance(response, AgentResponse) and isinstance(
                response.requires_tool, bool
            ):
                response = self._set_tool_metadata(response, tools_manager)
                return response
            raise ValueError("Invalid response structure")
        except Exception:
            # Fallback: plain LLM call + manual parsing
            raw = await self.llm.ainvoke(messages)
            content = raw.content if hasattr(raw, "content") else str(raw)
            return self._parse_agent_response(content, tools_manager)

    # ------------------------------------------------------------------
    # Step 1 — stream variant: yield raw tokens for direct answers
    # ------------------------------------------------------------------
    async def _step1_llm_define_tool_async(
        self,
        max_history: int = 5,
        user_id: str = "unknown_user",
        message: str = "",
        tools_manager: ToolManager = None,
        memory: Memory = None,
        skills: list = [],
        description: str = "",
        instruction: str = "",
        history: InConversationHistory = None,
        iteration: int = 1,
    ) -> AgentResponse:
        """
        Step 1: Build prompt and invoke LLM to get AgentResponse.
        On the first iteration, initializes conversation history with system + user messages.
        On subsequent iterations, appends the updated query as a new HumanMessage.
        """
        _history = self._preprocessing_messages(
            iteration=iteration,
            max_history=max_history,
            user_id=user_id,
            message=message,
            tools_manager=tools_manager,
            memory=memory,
            skills=skills,
            description=description,
            instruction=instruction,
            history=history,
        )
        return await self.define_tools_async(
            messages=_history, tools_manager=tools_manager
        )

    # ------------------------------------------------------------------
    # Step 2 — tool execution (no streaming, pure async I/O)
    # ------------------------------------------------------------------

    async def _step2_tool_invoke_async(
        self,
        current_query: str,
        response: AgentResponse,
        tools_manager: ToolManager,
        history: InConversationHistory,
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
        previous_prompt_tool: PromptToolResult = None,
        iteration: int = 1,
    ) -> tuple[str, ToolMessage | None, bool]:
        """
        Async variant of Step 2: execute the tool call (or fix_bug_command)
        from the ``AgentResponse``.

        Returns:
            current_query (str): Updated query for the next iteration.
            tool_message (ToolMessage | None): Result of tool execution.
            should_continue (bool): True if the loop should proceed.
        """
        # --- 2a. Handle fix_bug_command ---
        fix_cmd = getattr(response, "fix_bug_command", None)
        if fix_cmd:
            next_query, fix_msg, should_continue = self._handle_fix_bug_command(
                fix_cmd=fix_cmd, query=current_query, response=response, history=history
            )
            return next_query, fix_msg, should_continue, previous_prompt_tool

        # --- 2b. No tool call — direct answer ---
        if not getattr(response, "requires_tool", False) or not getattr(
            response, "tool_call", None
        ):
            return current_query, None, False, previous_prompt_tool

        # --- 2c. Validate tool data ---
        tool_data = response.tool_call.model_dump()
        if not tool_data:
            logger.warning(
                "LLM generated empty or invalid tool_call. Prompting for correction."
            )
            history.add_message(AIMessage(content=str(response)))
            return (
                "Your response was invalid. You must provide either a valid `tool_call`, "
                "a valid `answer`, or a `fix_bug_command`.",
                None,
                True,
                previous_prompt_tool,
            )

        # --- 2d. Guardrail check ---
        logger.info(f"Executing async-stream tool call: {tool_data}")
        try:
            is_valid_tool_permission = self.guardrail_executor.check_tool_guardrail(
                llm=self.llm,
                tool_name=tool_data.get("tool_name"),
                user_input=current_query,
            )
        except Exception:
            is_valid_tool_permission = False

        # --- 2e. Wrap AIMessage with tool_calls metadata ---
        # Generate a fresh ID per invocation so AIMessage and ToolMessage
        # always share the same tool_call_id, regardless of the static
        # registry UUID or any LLM-generated ID.
        _all_tools = tools_manager.load_tools()
        # Patch tool_data with the authoritative registry tool_call_id so
        # tool_data is always consistent before it flows into the adapter.
        _registry_id = _all_tools.get(tool_data.get("tool_name", ""), {}).get(
            "tool_call_id"
        )
        if _registry_id:
            tool_data["tool_call_id"] = _registry_id
        invocation_id = "tool_" + str(uuid.uuid4())
        content_val = response.answer if getattr(response, "answer", None) else ""
        try:
            ai_message = adapter_ai_response_with_tool_calls(
                tools_manager.load_tools(),
                AIMessage(content=content_val, iteration_id=iteration),
                tool_data,
                tool_call_id=invocation_id,
            )
            history.add_message(ai_message, iteration_id=iteration, is_rearrange=True)

        except ValueError as e:
            logger.warning(str(e))
            # The agent hallucinated a tool. Don't crash, feed the error back.
            fake_tool_call = {
                "name": tool_data.get("tool_name", "unknown_tool"),
                "args": tool_data.get("arguments", {}),
                "id": invocation_id,
                "type": tool_data.get("tool_type", "function"),
            }
            ai_msg_failed = AIMessage(content=content_val, iteration_id=iteration)
            ai_msg_failed.tool_calls = [fake_tool_call]
            history.add_message(
                ai_msg_failed,
                iteration_id=iteration,
                is_rearrange=True,
            )
            error_msg = ToolMessage(
                content=str(e),
                tool_call_id=invocation_id,
                additional_kwargs={"is_error": True},
                iteration_id=iteration,
            )
            history.add_message(error_msg, iteration_id=iteration, is_rearrange=True)
            # On tool failure: keep the original query unchanged so the LLM retries
            return current_query, error_msg, True, previous_prompt_tool

        # --- 2f. Execute tool asynchronously ---
        if is_valid_tool_permission:
            tool_message = await tools_manager._execute_tool(
                tool_name=tool_data["tool_name"],
                tool_type=tool_data["tool_type"],
                arguments=tool_data["arguments"],
                module_path=tool_data["module_path"],
                mcp_client=mcp_client,
                mcp_server_name=mcp_server_name,
                seconds_limit=self.seconds_limit,
            )
            if tool_message is None:
                tool_message = ToolMessage(
                    content="Tool execution success without artifact",
                    additional_kwargs={"is_error": False},
                    tool_call_id=invocation_id,
                    iteration_id=iteration,
                )
            else:
                # Override the static registry ID with the per-invocation ID
                # so it always matches ai_message.tool_calls[0]['id'].
                tool_message.tool_call_id = invocation_id
        else:
            tool_message = ToolMessage(
                content="Tool is not permitted by security rules.",
                additional_kwargs={"is_error": True},
                tool_call_id=invocation_id,
                iteration_id=iteration,
            )

        history.add_message(tool_message, iteration_id=iteration, is_rearrange=True)

        return current_query, tool_message, True, previous_prompt_tool

    # ------------------------------------------------------------------
    # Step 3 — stream final LLM summarisation
    # ------------------------------------------------------------------

    async def _step3_final_response_astream(
        self,
        query: str,
        tool_message: ToolMessage | None,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
        history: InConversationHistory = None,
        memory: Memory = None,
        user_id: str = None,
        iteration: int = 1,
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """
        Step 3 (async stream variant): Stream or yield the final response
        after the tool-calling loop ends.

        If ``is_tool_formatted=True``, asks the LLM to summarise tool
        results and yields chunks in real time via ``llm.astream()``.
        Otherwise yields the raw ``tool_message`` directly.

        Yields:
            AIMessageChunk | ToolMessage: Streamed chunks or raw tool message.
        """
        if is_tool_formatted:
            history.add_message(
                HumanMessage(
                    content=f"Based on the previous tool executions, please provide a final response to: {query}",
                    iteration_id=iteration,
                )
            )
            _history = history.get_history(max_history=max_history)
            _history = self._sanitize_history(_history)
            full_content = AIMessageChunk(content="")
            async for chunk in self.llm.astream(_history):
                full_content += chunk
                yield chunk
            full_content.iteration_id = iteration
            history.add_message(full_content, iteration_id=iteration)
            if memory and is_save_memory:
                memory.save_memory(message=full_content.content, user_id=user_id)
        else:
            self.guardrail_executor.check_output_guardrail(tool_message)
            if memory and is_save_memory:
                content = (
                    tool_message.content
                    if hasattr(tool_message, "content")
                    else str(tool_message)
                )
                memory.save_memory(message=content, user_id=user_id)
            yield tool_message
