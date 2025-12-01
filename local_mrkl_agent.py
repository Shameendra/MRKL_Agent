#!/usr/bin/env python3
"""
Simple Local MRKL Agent - No API Required!
==========================================

A minimal MRKL (Modular Reasoning, Knowledge and Language) agent
that uses rule-based reasoning instead of an LLM.

This demonstrates the MRKL architecture pattern:
- Multiple specialized tools
- Router to select the right tool
- Thought → Action → Observation loop

No API keys needed! Runs completely locally.
"""

import re
import math
from datetime import datetime, timedelta
from typing import Dict, Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Response from the agent"""
    thought: str
    action: str
    tool_used: str
    observation: str
    final_answer: str


class LocalMRKLAgent:
    """
    A simple MRKL agent that routes queries to specialized tools.
    Uses pattern matching instead of LLM for routing.
    """
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools"""
        
        # Calculator
        self.register_tool(
            name="calculator",
            func=self._calculator,
            description="Math calculations: +, -, *, /, ^, sqrt, sin, cos, etc."
        )
        
        # Date/Time
        self.register_tool(
            name="datetime",
            func=self._datetime_tool,
            description="Date and time operations, days between dates"
        )
        
        # Unit Converter
        self.register_tool(
            name="converter",
            func=self._unit_converter,
            description="Convert units: km/miles, kg/lbs, C/F, EUR/USD"
        )
        
        # Dictionary/Definitions
        self.register_tool(
            name="dictionary",
            func=self._dictionary,
            description="Word definitions and explanations"
        )
        
        # Tip Calculator
        self.register_tool(
            name="tip_calculator",
            func=self._tip_calculator,
            description="Calculate tips and split bills"
        )
    
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a new tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = description
    
    def run(self, query: str) -> AgentResponse:
        """
        Process a query using the MRKL pattern:
        Thought → Action → Observation → Answer
        """
        query_lower = query.lower()
        
        # Step 1: THOUGHT - Analyze the query
        thought, tool_name = self._route_query(query_lower)
        
        # Step 2: ACTION - Execute the tool
        if tool_name and tool_name in self.tools:
            action = f"Using tool: {tool_name}"
            observation = self.tools[tool_name](query)
        else:
            action = "No specific tool needed"
            observation = self._general_response(query)
        
        # Step 3: Generate final answer
        final_answer = self._format_answer(observation, tool_name)
        
        return AgentResponse(
            thought=thought,
            action=action,
            tool_used=tool_name or "none",
            observation=observation,
            final_answer=final_answer
        )
    
    def _route_query(self, query: str) -> Tuple[str, Optional[str]]:
        """Route query to appropriate tool based on patterns"""
        
        # Tip calculator patterns (check FIRST - higher priority)
        tip_patterns = [
            r'(tip|split|bill)',
            r'\d+.*tip',
            r'(how much|calculate).*tip',
        ]
        for pattern in tip_patterns:
            if re.search(pattern, query):
                return "This requires tip/bill calculation.", "tip_calculator"
        
        # DateTime patterns (check before calculator)
        date_patterns = [
            r"(what|current).*(time|date|day)",
            r"today'?s?\s*(date|day)?",
            r'(tomorrow|yesterday)',
            r'(days|weeks|months).*(between|until|since)',
            r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',
        ]
        for pattern in date_patterns:
            if re.search(pattern, query):
                return "This requires date/time information.", "datetime"
        
        # Dictionary patterns (check before calculator for "what is X")
        dict_patterns = [
            r'what is (mrkl|rag|llm|faiss|langchain|embedding|transformer|agent)',
            r'(define|meaning|definition).*(of|means)',
            r'(explain|describe)\s+(mrkl|rag|llm)',
        ]
        for pattern in dict_patterns:
            if re.search(pattern, query):
                return "This requires a definition lookup.", "dictionary"
        
        # Converter patterns
        convert_patterns = [
            r'convert\s+\d+',
            r'\d+\s*(km|miles?|kg|lbs?|pounds?|celsius|fahrenheit|[°]?[cf])\b',
            r'\d+\s*(eur|usd|euro|dollar)',
            r'(how many|what is).*(in|to)\s*(km|miles|kg|lbs)',
            r'to (miles|km|celsius|fahrenheit|lbs|kg)',
        ]
        for pattern in convert_patterns:
            if re.search(pattern, query):
                return "This requires unit conversion.", "converter"
        
        # Calculator patterns (check LAST)
        calc_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',  # 5 + 3
            r'(calculate|compute|solve)\s+\d+',
            r'(sqrt|square root|sin|cos|tan)',
            r'(\d+\s*%\s*of\s*\d+)',  # percentage
        ]
        for pattern in calc_patterns:
            if re.search(pattern, query):
                return "This requires mathematical calculation.", "calculator"
        
        return "I'll try to answer this directly.", None
    
    # ============= TOOLS =============
    
    def _calculator(self, query: str) -> str:
        """Perform mathematical calculations"""
        try:
            query_clean = query.lower()
            
            # Handle percentage first
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', query_clean)
            if percent_match:
                pct, num = float(percent_match.group(1)), float(percent_match.group(2))
                result = (pct / 100) * num
                return f"{pct}% of {num} = {result}"
            
            # Handle sqrt
            sqrt_match = re.search(r'(?:sqrt|square root)\s*(?:of)?\s*(\d+(?:\.\d+)?)', query_clean)
            if sqrt_match:
                num = float(sqrt_match.group(1))
                return f"√{num} = {math.sqrt(num):.4f}"
            
            # Handle word-based operations
            query_clean = query_clean.replace('plus', '+')
            query_clean = query_clean.replace('minus', '-')
            query_clean = query_clean.replace('times', '*')
            query_clean = query_clean.replace('multiplied by', '*')
            query_clean = query_clean.replace('divided by', '/')
            query_clean = query_clean.replace('to the power of', '**')
            query_clean = query_clean.replace('^', '**')
            
            # Extract mathematical expression more carefully
            # First, try to find a clear math expression pattern
            expr_match = re.search(r'(\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^\%]\s*\d+(?:\.\d+)?)+)', query_clean)
            
            if expr_match:
                expr = expr_match.group(1).strip()
                expr = expr.replace('^', '**')
                # Safe eval
                result = eval(expr, {"__builtins__": {}}, {})
                return f"{expr} = {result}"
            
            # Try single number operations (like just "25 * 4")
            # Remove common words
            for word in ['what', 'is', 'calculate', 'compute', '?', 'equals']:
                query_clean = query_clean.replace(word, '')
            
            # Clean up and try to extract numbers and operators
            query_clean = query_clean.strip()
            
            # Pattern for basic math: number operator number (optionally more)
            basic_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)', query_clean)
            if basic_match:
                n1, op, n2 = basic_match.groups()
                expr = f"{n1} {op} {n2}"
                result = eval(expr, {"__builtins__": {}}, {})
                
                # Check for more operations
                remaining = query_clean[basic_match.end():]
                more_match = re.search(r'\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)', remaining)
                if more_match:
                    op2, n3 = more_match.groups()
                    expr = f"{n1} {op} {n2} {op2} {n3}"
                    result = eval(expr, {"__builtins__": {}}, {})
                
                return f"{expr} = {result}"
            
            return "Could not parse mathematical expression. Try: '25 * 4 + 10' or '15% of 200'"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def _datetime_tool(self, query: str) -> str:
        """Handle date and time queries"""
        now = datetime.now()
        query_lower = query.lower()
        
        if 'time' in query_lower:
            return f"Current time: {now.strftime('%H:%M:%S')}"
        
        if 'today' in query_lower or 'date' in query_lower:
            return f"Today is {now.strftime('%A, %B %d, %Y')}"
        
        if 'tomorrow' in query_lower:
            tomorrow = now + timedelta(days=1)
            return f"Tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}"
        
        if 'yesterday' in query_lower:
            yesterday = now - timedelta(days=1)
            return f"Yesterday was {yesterday.strftime('%A, %B %d, %Y')}"
        
        # Days until/since
        if 'days' in query_lower:
            date_match = re.search(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', query)
            if date_match:
                day, month, year = map(int, date_match.groups())
                if year < 100:
                    year += 2000
                target_date = datetime(year, month, day)
                diff = (target_date - now).days
                if diff > 0:
                    return f"{diff} days until {target_date.strftime('%B %d, %Y')}"
                else:
                    return f"{abs(diff)} days since {target_date.strftime('%B %d, %Y')}"
        
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _unit_converter(self, query: str) -> str:
        """Convert between units"""
        query_lower = query.lower()
        
        # Extract number
        num_match = re.search(r'(\d+(?:\.\d+)?)', query)
        if not num_match:
            return "Please provide a number to convert."
        
        value = float(num_match.group(1))
        
        # Distance
        if 'km' in query_lower and 'mile' in query_lower:
            if 'to mile' in query_lower or 'in mile' in query_lower:
                return f"{value} km = {value * 0.621371:.2f} miles"
            else:
                return f"{value} miles = {value * 1.60934:.2f} km"
        
        if 'km' in query_lower:
            return f"{value} km = {value * 0.621371:.2f} miles"
        
        if 'mile' in query_lower:
            return f"{value} miles = {value * 1.60934:.2f} km"
        
        # Weight
        if 'kg' in query_lower or 'kilogram' in query_lower:
            return f"{value} kg = {value * 2.20462:.2f} lbs"
        
        if 'lb' in query_lower or 'pound' in query_lower:
            return f"{value} lbs = {value * 0.453592:.2f} kg"
        
        # Temperature - check for explicit direction first
        if 'to c' in query_lower or 'to celsius' in query_lower:
            celsius = (value - 32) * 5/9
            return f"{value}°F = {celsius:.1f}°C"
        
        if 'to f' in query_lower or 'to fahrenheit' in query_lower:
            fahrenheit = (value * 9/5) + 32
            return f"{value}°C = {fahrenheit:.1f}°F"
        
        # If no explicit direction, guess from unit mentioned
        if 'fahrenheit' in query_lower or '°f' in query_lower:
            celsius = (value - 32) * 5/9
            return f"{value}°F = {celsius:.1f}°C"
        
        if 'celsius' in query_lower or '°c' in query_lower:
            fahrenheit = (value * 9/5) + 32
            return f"{value}°C = {fahrenheit:.1f}°F"
        
        # Currency (approximate rates)
        if 'eur' in query_lower or 'euro' in query_lower:
            if 'to usd' in query_lower or 'dollar' in query_lower:
                return f"€{value} ≈ ${value * 1.08:.2f} USD (approximate)"
        
        if 'usd' in query_lower or 'dollar' in query_lower:
            if 'to eur' in query_lower or 'euro' in query_lower:
                return f"${value} ≈ €{value * 0.93:.2f} EUR (approximate)"
        
        return f"Could not determine conversion type for: {query}"
    
    def _tip_calculator(self, query: str) -> str:
        """Calculate tips and split bills"""
        query_lower = query.lower()
        
        # Extract tip percentage first (default 15%)
        tip_match = re.search(r'(\d+)\s*%', query)
        tip_pct = float(tip_match.group(1)) if tip_match else 15
        
        # Extract number of people (default 1)
        people_match = re.search(r'(\d+)\s*(?:people|persons?|ways?|split)', query_lower)
        people = int(people_match.group(1)) if people_match else 1
        
        # Extract bill amount - look for number near currency or "bill"
        # Try to find amount after "on" (e.g., "tip on 85 euro")
        bill_match = re.search(r'(?:on|of|for)\s*(\d+(?:\.\d+)?)\s*(?:€|\$|euro|dollar)?', query_lower)
        
        if not bill_match:
            # Try to find amount before currency
            bill_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:€|\$|euro|dollar)', query_lower)
        
        if not bill_match:
            # Get the largest number that's not the tip percentage or people count
            numbers = re.findall(r'(\d+(?:\.\d+)?)', query)
            numbers = [float(n) for n in numbers]
            # Filter out tip percentage and people count
            numbers = [n for n in numbers if n != tip_pct and n != people and n > 5]
            if numbers:
                bill = max(numbers)  # Assume bill is the largest number
            else:
                return "Please provide the bill amount. Example: '20% tip on 85 euro split 4 ways'"
        else:
            bill = float(bill_match.group(1))
        
        tip = bill * (tip_pct / 100)
        total = bill + tip
        per_person = total / people
        
        result = f"""
Bill: €{bill:.2f}
Tip ({tip_pct:.0f}%): €{tip:.2f}
Total: €{total:.2f}"""
        
        if people > 1:
            result += f"\nPer person ({people} people): €{per_person:.2f}"
        
        return result.strip()
    
    def _dictionary(self, query: str) -> str:
        """Simple dictionary with common terms"""
        
        definitions = {
            "mrkl": "MRKL (Modular Reasoning, Knowledge and Language) is an AI architecture that combines LLMs with external tools/modules for enhanced reasoning.",
            "rag": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by retrieving relevant information from external sources.",
            "llm": "LLM (Large Language Model) is an AI model trained on vast text data to understand and generate human-like text.",
            "faiss": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
            "langchain": "LangChain is a framework for building applications powered by language models, providing tools for chains, agents, and memory.",
            "vector database": "A vector database stores data as high-dimensional vectors, enabling fast similarity searches for AI/ML applications.",
            "embedding": "An embedding is a numerical representation of data (text, images) in a continuous vector space where similar items are close together.",
            "transformer": "A transformer is a neural network architecture using self-attention mechanisms, the foundation of modern LLMs like GPT and BERT.",
            "agent": "An AI agent is a system that can perceive its environment, make decisions, and take actions to achieve goals autonomously.",
            "prompt engineering": "Prompt engineering is the practice of crafting effective inputs (prompts) to get desired outputs from language models.",
        }
        
        query_lower = query.lower()
        
        for term, definition in definitions.items():
            if term in query_lower:
                return f"**{term.upper()}**: {definition}"
        
        return "Term not found in dictionary. Try: MRKL, RAG, LLM, FAISS, LangChain, embedding, transformer, agent"
    
    def _general_response(self, query: str) -> str:
        """Fallback for queries that don't match any tool"""
        return f"I understood your query: '{query}'. However, I'm a simple rule-based agent without an LLM, so I can only help with: calculations, date/time, unit conversions, tips, and definitions."
    
    def _format_answer(self, observation: str, tool: Optional[str]) -> str:
        """Format the final answer"""
        if tool:
            return f"[{tool.upper()}] {observation}"
        return observation
    
    def list_tools(self) -> str:
        """List available tools"""
        output = ["🛠️ Available Tools:", "=" * 40]
        for name, desc in self.tool_descriptions.items():
            output.append(f"• {name}: {desc}")
        return "\n".join(output)


def main():
    """Interactive CLI for the local MRKL agent"""
    
    print("=" * 60)
    print("🤖 Local MRKL Agent - No API Required!")
    print("=" * 60)
    print()
    
    agent = LocalMRKLAgent()
    print(agent.list_tools())
    
    print("\n" + "=" * 60)
    print("Examples:")
    print("  • 'What is 25 * 4 + 10?'")
    print("  • 'Convert 100 km to miles'")
    print("  • 'Calculate 20% tip on €50 bill split 3 ways'")
    print("  • 'What is today's date?'")
    print("  • 'What is RAG?'")
    print("  • 'tools' to list tools, 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'tools':
                print(agent.list_tools())
                continue
            
            # Run agent
            response = agent.run(query)
            
            # Display MRKL reasoning chain
            print(f"\n💭 Thought: {response.thought}")
            print(f"🎬 Action: {response.action}")
            print(f"👁️ Observation: {response.observation}")
            print(f"\n✅ Answer: {response.final_answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
