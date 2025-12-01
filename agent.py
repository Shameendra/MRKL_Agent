"""
Multi-Tool MRKL Agent Implementation
MRKL: Modular Reasoning, Knowledge and Language
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import (
    OPENAI_API_KEY, LLM_MODEL, TEMPERATURE,
    MAX_ITERATIONS, MAX_EXECUTION_TIME, VERBOSE
)
from tools import get_all_tools, get_tool_descriptions

logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Possible agent actions"""
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    REASONING = "reasoning"


@dataclass
class ThoughtStep:
    """A single thought/action/observation step"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class AgentState:
    """State of the MRKL agent"""
    input: str
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    current_step: int = 0
    final_answer: Optional[str] = None
    error: Optional[str] = None


class MRKLAgent:
    """
    MRKL Agent: Modular Reasoning, Knowledge and Language
    
    This agent demonstrates:
    1. Tool selection based on task requirements
    2. Multi-step reasoning with thought chains
    3. Dynamic routing between tools
    4. Self-correction and error handling
    """
    
    SYSTEM_PROMPT = """You are a powerful AI assistant with access to multiple tools.

Available Tools:
{tool_descriptions}

To use a tool, respond with EXACTLY this format:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]

When you have enough information to answer, respond with:
Thought: [Your final reasoning]
Final Answer: [Your complete answer to the user]

Rules:
1. Always start with a Thought
2. Use tools when needed for accurate information
3. Break complex problems into steps
4. Verify calculations with the calculator tool
5. Use web_search for current information
6. If a tool fails, try an alternative approach
7. Always provide a Final Answer when done

Begin!"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        self.tools = {tool.name: tool for tool in get_all_tools()}
        self.tool_descriptions = get_tool_descriptions()
        self.max_iterations = MAX_ITERATIONS
    
    def _parse_response(self, response: str) -> Tuple[AgentAction, Dict[str, str]]:
        """Parse LLM response to extract action"""
        
        # Check for final answer
        if "Final Answer:" in response:
            match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            if match:
                return AgentAction.FINAL_ANSWER, {"answer": match.group(1).strip()}
        
        # Check for tool action
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+?)(?=Thought:|Action:|Final Answer:|$)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            
            return AgentAction.TOOL_CALL, {
                "thought": thought,
                "action": action,
                "action_input": action_input
            }
        
        # Just reasoning
        return AgentAction.REASONING, {"thought": thought}
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = tool.invoke(tool_input)
            return str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
    def _build_prompt(self, state: AgentState) -> str:
        """Build the prompt with conversation history"""
        history = f"User Question: {state.input}\n\n"
        
        for step in state.thought_steps:
            history += f"Thought: {step.thought}\n"
            if step.action:
                history += f"Action: {step.action}\n"
                history += f"Action Input: {step.action_input}\n"
            if step.observation:
                history += f"Observation: {step.observation}\n"
            history += "\n"
        
        return history
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a query
        
        Args:
            query: User question or task
        
        Returns:
            Dictionary with answer and reasoning trace
        """
        state = AgentState(input=query)
        
        # Format system prompt with tool descriptions
        system_prompt = self.SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions
        )
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # Build conversation
            prompt = self._build_prompt(state)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Get LLM response
            try:
                response = self.llm.invoke(messages)
                response_text = response.content
                
                if VERBOSE:
                    logger.info(f"\n--- Iteration {iteration} ---")
                    logger.info(f"Response: {response_text[:200]}...")
                
            except Exception as e:
                state.error = f"LLM error: {str(e)}"
                break
            
            # Parse response
            action_type, parsed = self._parse_response(response_text)
            
            if action_type == AgentAction.FINAL_ANSWER:
                state.final_answer = parsed["answer"]
                break
            
            elif action_type == AgentAction.TOOL_CALL:
                thought = parsed["thought"]
                action = parsed["action"]
                action_input = parsed["action_input"]
                
                # Execute tool
                observation = self._execute_tool(action, action_input)
                
                if VERBOSE:
                    logger.info(f"Tool: {action}")
                    logger.info(f"Input: {action_input}")
                    logger.info(f"Observation: {observation[:200]}...")
                
                # Record step
                state.thought_steps.append(ThoughtStep(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation
                ))
            
            elif action_type == AgentAction.REASONING:
                state.thought_steps.append(ThoughtStep(
                    thought=parsed.get("thought", "")
                ))
        
        # Check if we hit max iterations
        if not state.final_answer and not state.error:
            state.error = f"Max iterations ({self.max_iterations}) reached without final answer"
        
        return self._format_result(state)
    
    def _format_result(self, state: AgentState) -> Dict[str, Any]:
        """Format the final result"""
        return {
            "input": state.input,
            "answer": state.final_answer,
            "error": state.error,
            "steps": [
                {
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation
                }
                for step in state.thought_steps
            ],
            "iterations": len(state.thought_steps)
        }
    
    def stream(self, query: str):
        """Stream agent execution for real-time display"""
        state = AgentState(input=query)
        
        system_prompt = self.SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions
        )
        
        yield {"type": "start", "input": query}
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            prompt = self._build_prompt(state)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = self.llm.invoke(messages)
                response_text = response.content
            except Exception as e:
                yield {"type": "error", "message": str(e)}
                return
            
            action_type, parsed = self._parse_response(response_text)
            
            if action_type == AgentAction.FINAL_ANSWER:
                yield {"type": "final_answer", "answer": parsed["answer"]}
                return
            
            elif action_type == AgentAction.TOOL_CALL:
                yield {
                    "type": "thought",
                    "thought": parsed["thought"]
                }
                
                yield {
                    "type": "tool_call",
                    "tool": parsed["action"],
                    "input": parsed["action_input"]
                }
                
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                
                yield {
                    "type": "observation",
                    "observation": observation
                }
                
                state.thought_steps.append(ThoughtStep(
                    thought=parsed["thought"],
                    action=parsed["action"],
                    action_input=parsed["action_input"],
                    observation=observation
                ))
        
        yield {"type": "error", "message": "Max iterations reached"}


class MRKLAgentWithPlanning(MRKLAgent):
    """
    Extended MRKL Agent with explicit planning phase
    """
    
    PLANNING_PROMPT = """Given this task, create a step-by-step plan to solve it.

Task: {task}

Available Tools: {tools}

Create a numbered plan (3-7 steps) that:
1. Breaks down the task into subtasks
2. Identifies which tool to use for each step
3. Shows how information flows between steps

Plan:"""

    def plan(self, task: str) -> List[str]:
        """Generate an execution plan"""
        prompt = ChatPromptTemplate.from_template(self.PLANNING_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "task": task,
            "tools": list(self.tools.keys())
        })
        
        # Parse plan into steps
        steps = []
        for line in response.split("\n"):
            if re.match(r"^\d+\.", line.strip()):
                steps.append(line.strip())
        
        return steps
    
    def run_with_plan(self, query: str) -> Dict[str, Any]:
        """Run agent with explicit planning"""
        # First, create a plan
        plan = self.plan(query)
        
        if VERBOSE:
            logger.info("Generated Plan:")
            for step in plan:
                logger.info(f"  {step}")
        
        # Then execute normally
        result = self.run(query)
        result["plan"] = plan
        
        return result


# Convenience function
def ask_agent(question: str, with_planning: bool = False) -> Dict[str, Any]:
    """
    Simple interface to query the MRKL agent
    
    Args:
        question: User question
        with_planning: Whether to use planning phase
    
    Returns:
        Agent result with answer and reasoning trace
    """
    if with_planning:
        agent = MRKLAgentWithPlanning()
        return agent.run_with_plan(question)
    else:
        agent = MRKLAgent()
        return agent.run(question)
