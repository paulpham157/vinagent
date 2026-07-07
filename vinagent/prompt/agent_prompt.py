from __future__ import annotations
from typing import Union, List, Optional
from dataclasses import dataclass, field
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from vinagent.memory import Memory
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptToolResult:
    prompt: str
    tool_call: str
    tool_message: ToolMessage
    history: list[BaseMessage]
    prior_fix_str: str
    previous: Optional[PromptToolResult] = None

    def format_tool_template(
        self,
        query: str,
        tool_call: str,
        full_result: str,
        prior_fix_str: str,
        previous: Optional[PromptToolResult] = None,
    ):
        query_info, tool_info, full_result, prior_fix_str = None, None, None, None
        if previous:
            query_info = f"- Question: {query}\n"
            tool_info = f"- Tool Used: {tool_call}\n" if tool_call else ""
            full_result_info = (
                f"""- Tool's Result:\n{full_result}\n""" if full_result else ""
            )
            prior_fix_str_info = (
                f"{self.action_prompt(prior_fix_str)}" if prior_fix_str else ""
            )
        else:
            query_info = f"- Question: {query}\n"
            tool_info = f"- Tool Used: {tool_call}\n" if tool_call else ""
            full_result_info = (
                f"""- Tool's Result:\n{full_result}\n""" if full_result else ""
            )
            prior_fix_str_info = (
                f"{self.action_prompt(prior_fix_str)}" if prior_fix_str else ""
            )

        tool_template = query_info + tool_info + full_result_info + prior_fix_str_info
        return tool_template

    def action_prompt(self, prior_fix_str: str) -> str:
        _action_prompt = (
            "\n\n[TOOL ERROR RECOVERY REQUIRED]\n"
            "The previous tool call FAILED with the error shown above.\n"
            f"{prior_fix_str}\n\n"
            "You must FIRST analyze the root cause of the error.\n\n"
            "There are TWO possible recovery strategies:\n\n"
            "Step 1. CODE BUG (inside the tool command)\n"
            "- The Python code is incorrect\n"
            "- A runtime exception occurred (TypeError, ValueError, AttributeError, etc.)\n"
            "- Incorrect API usage\n"
            "- Logic bug in the script\n"
            "Action:\n"
            "  - Modify the Python code inside `tool_call.command`\n"
            "  - Set `requires_tool: True`\n"
            "  - Do NOT use `fix_bug_command`\n\n"
            "Step 2. ENVIRONMENT ISSUE\n"
            "- Missing library\n"
            "- Missing binary\n"
            "- PATH problem\n"
            "- Permission problem\n"
            "➡ Action:\n"
            "  - Provide a bash script in `fix_bug_command`\n"
            "  - Set `requires_tool: False`\n"
            "  - Do NOT modify the tool code\n\n"
            "IMPORTANT RULES:\n"
            "- Python exceptions almost always mean a CODE BUG.\n"
            "- Only install packages if the error explicitly says 'ModuleNotFoundError' or 'command not found'.\n"
            "- Do NOT repeat previous fixes.\n"
            "- The generated Python script MUST run the true procedure (not just idea/prompt to run the procedure)\n"
            "- Print the final result.\n"
            "- If creating output artifacts, print the artifact paths.\n"
            "- Scripts must never produce empty stdout.\n"
            "Respond in the structured JSON format with ONE of the two strategies."
        )
        return _action_prompt


class PromptHandler:
    def format_tools_as_xml(self, tools: dict) -> str:
        parts = ["<tools>"]
        for name, tool in tools.items():
            parts.append(f'  <tool name="{name}">')
            parts.append(f'    <tool_name>{tool["tool_name"]}</tool_name>')
            parts.append(f'    <tool_type>{tool["tool_type"]}</tool_type>')
            parts.append(f'    <module_path>{tool["module_path"]}</module_path>')
            parts.append(f'    <is_runtime>{tool["is_runtime"]}</is_runtime>')
            parts.append(f'    <arguments>{json.dumps(tool["arguments"])}</arguments>')
            parts.append(f'    <return_type>{tool["return"]}</return_type>')
            parts.append(f'    <docstring>\n{tool["docstring"]}\n    </docstring>')
            parts.append(f"  </tool>")
        parts.append("</tools>")
        return "\n".join(parts)

    def action_prompt(self, prior_fix_str: str) -> str:
        _action_prompt = (
            "⚠️ CRITICAL: The previous tool call FAILED with an error.\n"
            "DO NOT OUTPUT THE EXACT SAME CODE THAT ALREADY FAILED! YOU MUST FIX THE BUG.\n"
            f"{prior_fix_str}\n\n"
            "You must FIRST analyze the root cause of the error.\n\n"
            "There are TWO possible recovery strategies:\n\n"
            "Step 1. CODE BUG (inside the tool command)\n"
            "- The Python code is incorrect\n"
            "- A runtime exception occurred (TypeError, ValueError, AttributeError, etc.)\n"
            "- Incorrect API usage\n"
            "- Logic bug in the script\n"
            "Action:\n"
            "  - Modify the Python code inside `tool_call.command`\n"
            "  - Set `requires_tool: True`\n"
            "  - Do NOT use `fix_bug_command`\n\n"
            "Step 2. ENVIRONMENT ISSUE\n"
            "- Missing library\n"
            "- Missing binary\n"
            "- PATH problem\n"
            "- Permission problem\n"
            "➡ Action:\n"
            "  - Provide a bash script in `fix_bug_command`\n"
            "  - Set `requires_tool: False`\n"
            "  - Do NOT modify the tool code\n\n"
            "IMPORTANT RULES:\n"
            "- Python exceptions almost always mean a CODE BUG.\n"
            "- Only install packages if the error explicitly says 'ModuleNotFoundError' or 'command not found'.\n"
            "- Do NOT repeat previous fixes.\n"
            "- The generated Python script MUST run the true procedure (not just idea/prompt to run the procedure)\n"
            "- Print the final result.\n"
            "- If creating output artifacts, print the artifact paths.\n"
            "- Scripts must never produce empty stdout.\n"
        )
        return _action_prompt

    def build_prompt(
        self,
        user_id: str,
        message: str,
        tools: dict,
        memory: Memory,
        history: Optional[list[BaseMessage]] = None,
    ) -> str:
        memory_content = ""
        if memory:
            memory_content = memory.load_memory_by_user(
                load_type="string", user_id=user_id
            )

        _error_instruction = ""
        _success_instruction = ""
        if history and len(history) > 0:
            last_msg = history[-1]
            if isinstance(last_msg, ToolMessage):
                if last_msg.additional_kwargs.get("is_error", False):
                    prior_fixes = []
                    for i, msg in enumerate(history):
                        if isinstance(msg, ToolMessage) and msg.additional_kwargs.get(
                            "is_error"
                        ):
                            if (i >= 1) and isinstance(history[i - 1], AIMessage):
                                tc = (
                                    history[i - 1].tool_calls[0]
                                    if getattr(history[i - 1], "tool_calls", None)
                                    else {}
                                )
                                cmd = (
                                    (tc.get("args") or {}).get("command", "")
                                    if isinstance(tc, dict)
                                    else ""
                                )
                                cmd_str = str(cmd)
                                if len(cmd_str) > 200:
                                    cmd_str = cmd_str[:200] + "... [TRUNCATED]"
                                prior_fixes.append(f"  - (Tried code) {cmd_str}")

                                err_content = str(msg.content)
                                if len(err_content) > 500:
                                    err_content = err_content[:500] + "... [TRUNCATED]"
                                prior_fixes.append(
                                    f"  - (Resulting error) {err_content}\n"
                                )

                    prior_fix_str = ""
                    if prior_fixes:
                        prior_fix_str = (
                            "\nPrevious fix attempts (already tried, do NOT repeat):\n"
                            + "\n".join(prior_fixes)
                        )

                    _error_instruction = self.action_prompt(prior_fix_str)
                else:
                    _success_instruction = (
                        "[TOOL EXECUTED SUCCESSFULLY]\n"
                        "Check the tool's result in the conversation history.\n\n"
                        "If the overall task requested by the user is FULLY complete: you MUST output a standard JSON response with `requires_tool`: false and provide a summary `answer` to the user. Do NOT call the same tool again.\n\n"
                        "If there are STILL REMAINING STEPS to finish the overall task: you MUST output `requires_tool`: true and call the NEXT tool necessary to complete the task."
                    )

        prompt = f"""You are a smart assistant that can answer questions and use tools to complete tasks.
        
## User
{user_id}

## Memory
{memory_content.replace("I ", f"{user_id} ") if memory_content else "No memory available."}

## Task
{message}

## Available Tools
{self.format_tools_as_xml(tools)}

---

## Instructions

### Step 1 — Decide if a tool is needed
- If the task is casual conversation or a simple factual question, answer directly as a helpful assistant. Do NOT invoke a tool.
- For multi-step tasks, if you have successfully executed a tool but there are STILL MORE steps left, you MUST call the NEXT tool. Do NOT stop early.
- If the ENTIRE overall task is fully complete, DO NOT call any more tools. Instead, answer the user directly with the final status.
- If the task requires file creation, data processing, or document manipulation and hasn't been done yet, select the most appropriate tool from the list above.

### Step 2 — If a tool IS needed, output ONLY this JSON (no explanation, no markdown). For example:
{{
"tool_name": "<exact tool name from the list>",
"tool_type": "<one of: function | module | mcp | agentskills>",
"module_path": "<module_path from the tool definition>",
"is_runtime": <true or false>,
"tool_call_id": "<tool_call_id from the tool definition>",
"arguments": {{
    "command": "<see rules below>"
}},
"return": "<return type from the tool definition>"
}}

### Step 3 — How to fill in `command`

The `command` value depends on `tool_type`:

| tool_type | What to write in `command` |
|---|---|
| `agentskills` | A **complete, agent-contained, runnable Python or bash script** that solves the user's task. Follow the code patterns in the tool's docstring. Save outputs to `/mnt/user-data/outputs/`. Do NOT copy docstring examples verbatim — write code specific to the task. |
| `function` | A function call string, e.g. `"get_weather(city='Hanoi')"` |
| `module` | A module-level invocation string appropriate for that module |
| `mcp` | The MCP action string as specified by that tool |

### Step 4 - Tool Execution Feedback
{_error_instruction or _success_instruction}

### Rules
- Never make up tool names, module paths, or arguments not present in the Available Tools.
- Never mix syntax from one tool into another.
- If you are unsure which tool to use, answer directly and suggest where the user might find help.
- Do not add explanation or commentary when outputting a tool call — output JSON only.
"""
        return prompt

    def system_prompt(
        self, skills: list[str], description: str, instruction: str, iteration: int = 1
    ) -> str:
        skills = "- " + "- ".join(skills)
        content = (
            f"{description}\nYour skills:\n{skills}\nInstruction:\n{instruction}"
            if instruction
            else f"{description}\nYour skills:\n{skills}"
        )
        system_prompt = SystemMessage(content=content, iteration_id=iteration)
        return system_prompt

    def prompt_tool(
        self,
        query: str,
        tool_call: str,
        tool_message: ToolMessage,
        history: list[BaseMessage],
        *args,
        previous: Optional[PromptToolResult] = None,
        **kwargs,
    ) -> str:
        # Check if the tool execution resulted in an error
        is_error = tool_message.additional_kwargs.get("is_error", False)

        # Include BOTH the summary content AND the full artifact (STDERR traceback)
        # so the LLM has the complete picture to generate a comprehensive fix.
        content_value = tool_message.content or ""
        artifact_value = getattr(tool_message, "artifact", None)

        import pandas as pd

        if isinstance(artifact_value, pd.DataFrame):
            artifact_value = artifact_value.to_string()
        else:
            artifact_value = artifact_value or ""

        # Now it is safe to compare
        if artifact_value != "" and artifact_value != content_value:
            full_result = f"{content_value}\n\nFull output:\n{artifact_value}"
        else:
            full_result = content_value or artifact_value

        prior_fix_str = ""
        if is_error:
            # Collect prior fix attempts from history for context
            prior_fixes = []

            for msg in history:
                if (
                    isinstance(msg, ToolMessage)
                    and msg.additional_kwargs.get("is_error") is False
                ):
                    # successful fix attempt
                    prior_fixes.append(f"  - (success) {msg.content}")
                elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        name = (
                            tc.get("name")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", None)
                        )
                        if name == "fix_bug_command":
                            cmd = (
                                (tc.get("args") or {}).get("command", "")
                                if isinstance(tc, dict)
                                else ""
                            )
                            prior_fixes.append(f"  - (tried) {cmd}")
            if prior_fixes:
                prior_fix_str = (
                    "\n\nPrevious fix attempts (already tried, do NOT repeat):\n"
                    + "\n".join(prior_fixes)
                )
        else:
            prior_fix_str = (
                "The tool executed SUCCESSFULLY. Please review the Tool's Result above.\n"
                "If the task is complete, you MUST output a standard JSON response with `requires_tool`: false and provide a summary `answer` to the user. Do NOT call the same tool again."
            )

        # tool_template = (
        #     f"- Question: {query}\n"
        #     f"- Tool Used: {tool_call}\n"
        #     f"""- Tool's Result:\n{full_result}\n"""
        #     f"{_action_prompt}"
        # )
        prompt_tool_result = PromptToolResult(
            prompt=query,
            tool_call=tool_call,
            tool_message=full_result,
            history=history,
            prior_fix_str=prior_fix_str,
            previous=previous,
        )
        tool_template = prompt_tool_result.format_tool_template(
            query=query,
            tool_call=tool_call,
            full_result=full_result,
            prior_fix_str=prior_fix_str,
            previous=previous,
        )
        return tool_template, prompt_tool_result
