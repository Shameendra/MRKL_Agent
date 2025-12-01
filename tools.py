"""
Tool Implementations for MRKL Agent
Diverse tools for different reasoning tasks
"""

import os
import json
import re
import math
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests
from abc import ABC, abstractmethod
import logging

from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from config import (
    OPENAI_API_KEY, SERPAPI_KEY, WOLFRAM_APP_ID, 
    NEWS_API_KEY, TOOL_TIMEOUT
)

logger = logging.getLogger(__name__)


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

class SearchInput(BaseModel):
    query: str = Field(description="The search query")


@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this when you need up-to-date information or facts you don't know.
    """
    if not SERPAPI_KEY:
        return "Web search is not configured. Please provide SERPAPI_KEY."
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        
        response = requests.get(url, params=params, timeout=TOOL_TIMEOUT)
        data = response.json()
        
        results = []
        
        # Organic results
        for item in data.get("organic_results", [])[:3]:
            results.append(f"- {item.get('title', '')}: {item.get('snippet', '')}")
        
        # Answer box if available
        if "answer_box" in data:
            answer = data["answer_box"].get("answer") or data["answer_box"].get("snippet")
            if answer:
                results.insert(0, f"Quick Answer: {answer}")
        
        return "\n".join(results) if results else "No results found."
        
    except Exception as e:
        return f"Search error: {str(e)}"


# =============================================================================
# CALCULATOR TOOL
# =============================================================================

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    Use this for any mathematical calculations.
    Supports: +, -, *, /, ^, sqrt, sin, cos, tan, log, exp, pi, e
    """
    try:
        # Sanitize and prepare expression
        expr = expression.replace("^", "**")
        expr = expr.replace("sqrt", "math.sqrt")
        expr = expr.replace("sin", "math.sin")
        expr = expr.replace("cos", "math.cos")
        expr = expr.replace("tan", "math.tan")
        expr = expr.replace("log", "math.log")
        expr = expr.replace("exp", "math.exp")
        expr = expr.replace("pi", str(math.pi))
        expr = expr.replace(" e ", f" {math.e} ")
        
        # Safe evaluation
        allowed_chars = set("0123456789+-*/().math sqrtsincoantlogexp ")
        if not all(c in allowed_chars or c.isalpha() for c in expr):
            return "Invalid characters in expression"
        
        result = eval(expr, {"__builtins__": {}, "math": math})
        
        return f"Result: {result}"
        
    except Exception as e:
        return f"Calculation error: {str(e)}"


# =============================================================================
# WOLFRAM ALPHA TOOL (Advanced Math/Science)
# =============================================================================

class WolframInput(BaseModel):
    query: str = Field(description="Question for Wolfram Alpha")


@tool("wolfram_alpha", args_schema=WolframInput)
def wolfram_alpha(query: str) -> str:
    """
    Query Wolfram Alpha for complex calculations, scientific data, and factual information.
    Use this for: advanced math, unit conversions, scientific constants, statistics, etc.
    """
    if not WOLFRAM_APP_ID:
        return "Wolfram Alpha is not configured. Please provide WOLFRAM_APP_ID."
    
    try:
        url = "http://api.wolframalpha.com/v2/query"
        params = {
            "input": query,
            "appid": WOLFRAM_APP_ID,
            "format": "plaintext",
            "output": "json"
        }
        
        response = requests.get(url, params=params, timeout=TOOL_TIMEOUT)
        data = response.json()
        
        if data.get("queryresult", {}).get("success"):
            pods = data["queryresult"].get("pods", [])
            results = []
            
            for pod in pods[:3]:
                title = pod.get("title", "")
                subpods = pod.get("subpods", [])
                for subpod in subpods:
                    plaintext = subpod.get("plaintext", "")
                    if plaintext:
                        results.append(f"{title}: {plaintext}")
            
            return "\n".join(results) if results else "No results from Wolfram Alpha."
        else:
            return "Wolfram Alpha couldn't process this query."
            
    except Exception as e:
        return f"Wolfram Alpha error: {str(e)}"


# =============================================================================
# CODE EXECUTOR TOOL
# =============================================================================

class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute")


@tool("python_executor", args_schema=CodeInput)
def python_executor(code: str) -> str:
    """
    Execute Python code safely.
    Use this when you need to run calculations, data processing, or algorithms.
    Note: Has limited capabilities for security.
    """
    # Safe execution environment
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "len": len,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "zip": zip,
        "enumerate": enumerate,
        "map": map,
        "filter": filter,
    }
    
    try:
        # Capture output
        import io
        import sys
        
        output_buffer = io.StringIO()
        
        # Execute code
        exec_globals = safe_globals.copy()
        exec(code, exec_globals)
        
        # Get result
        result = exec_globals.get("result", None)
        
        if result is not None:
            return f"Result: {result}"
        else:
            return "Code executed successfully (no result variable set)"
            
    except Exception as e:
        return f"Execution error: {str(e)}"


# =============================================================================
# NEWS TOOL
# =============================================================================

class NewsInput(BaseModel):
    topic: str = Field(description="News topic to search for")


@tool("news_search", args_schema=NewsInput)
def news_search(topic: str) -> str:
    """
    Search for recent news articles on a topic.
    Use this for current events and recent news.
    """
    if not NEWS_API_KEY:
        # Fallback to web search
        return web_search(f"{topic} news today")
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "apiKey": NEWS_API_KEY,
            "sortBy": "publishedAt",
            "pageSize": 5
        }
        
        response = requests.get(url, params=params, timeout=TOOL_TIMEOUT)
        data = response.json()
        
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            results = []
            
            for article in articles[:5]:
                title = article.get("title", "")
                source = article.get("source", {}).get("name", "")
                description = article.get("description", "")[:150]
                results.append(f"- [{source}] {title}: {description}...")
            
            return "\n".join(results) if results else "No news found."
        else:
            return f"News API error: {data.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"News search error: {str(e)}"


# =============================================================================
# DATABASE QUERY TOOL (SQL)
# =============================================================================

class SQLInput(BaseModel):
    query: str = Field(description="SQL query to execute")


@tool("sql_query", args_schema=SQLInput)
def sql_query(query: str) -> str:
    """
    Execute SQL queries against the knowledge database.
    Use this when you need to query structured data.
    Available tables: products, customers, orders, employees
    """
    import sqlite3
    
    # Create in-memory demo database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            country TEXT
        )
    """)
    
    # Insert sample data
    products = [
        (1, "Laptop", "Electronics", 999.99, 50),
        (2, "Smartphone", "Electronics", 699.99, 100),
        (3, "Headphones", "Electronics", 149.99, 200),
        (4, "Desk Chair", "Furniture", 249.99, 30),
        (5, "Monitor", "Electronics", 399.99, 75),
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)
    
    try:
        # Only allow SELECT queries for safety
        if not query.strip().upper().startswith("SELECT"):
            return "Only SELECT queries are allowed for security reasons."
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        # Format results
        output = [", ".join(columns)]
        for row in results[:10]:
            output.append(", ".join(str(v) for v in row))
        
        conn.close()
        return "\n".join(output) if results else "No results found."
        
    except Exception as e:
        conn.close()
        return f"SQL error: {str(e)}"


# =============================================================================
# DATE/TIME TOOL
# =============================================================================

class DateTimeInput(BaseModel):
    query: str = Field(description="Date/time question or calculation")


@tool("datetime_tool", args_schema=DateTimeInput)
def datetime_tool(query: str) -> str:
    """
    Handle date and time queries.
    Use for: current time, date calculations, timezone conversions, etc.
    Examples: "current time", "days until Christmas", "2 weeks from now"
    """
    now = datetime.now()
    query_lower = query.lower()
    
    try:
        if "current" in query_lower or "now" in query_lower or "today" in query_lower:
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        elif "days until" in query_lower or "days to" in query_lower:
            # Try to extract target date
            if "christmas" in query_lower:
                target = datetime(now.year, 12, 25)
                if target < now:
                    target = datetime(now.year + 1, 12, 25)
            elif "new year" in query_lower:
                target = datetime(now.year + 1, 1, 1)
            else:
                return "Please specify a known date or use format YYYY-MM-DD"
            
            delta = (target - now).days
            return f"Days until {target.strftime('%B %d, %Y')}: {delta} days"
        
        elif "weeks from now" in query_lower or "days from now" in query_lower:
            # Extract number
            numbers = re.findall(r'\d+', query)
            if numbers:
                num = int(numbers[0])
                if "week" in query_lower:
                    future = now + timedelta(weeks=num)
                else:
                    future = now + timedelta(days=num)
                return f"{num} {'weeks' if 'week' in query_lower else 'days'} from now: {future.strftime('%Y-%m-%d')}"
        
        elif "day of week" in query_lower:
            return f"Today is {now.strftime('%A')}"
        
        return f"Current date: {now.strftime('%Y-%m-%d')}, Time: {now.strftime('%H:%M:%S')}"
        
    except Exception as e:
        return f"Date/time error: {str(e)}"


# =============================================================================
# KNOWLEDGE RETRIEVAL TOOL (RAG)
# =============================================================================

class KnowledgeInput(BaseModel):
    query: str = Field(description="Question to search in knowledge base")


@tool("knowledge_base", args_schema=KnowledgeInput)
def knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for domain-specific information.
    Use this when you need to find information from documents, manuals, or FAQs.
    """
    # Simulated knowledge base (in production, this would be a real RAG system)
    knowledge = {
        "company policy": "Our company follows a flexible work policy with hybrid options. Standard hours are 9 AM to 5 PM.",
        "refund policy": "Customers can request refunds within 30 days of purchase. Digital products are non-refundable after download.",
        "technical support": "Technical support is available 24/7 via email at support@company.com or phone at 1-800-SUPPORT.",
        "product warranty": "All products come with a 1-year limited warranty covering manufacturing defects.",
        "shipping": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available for an additional fee.",
    }
    
    query_lower = query.lower()
    results = []
    
    for topic, info in knowledge.items():
        if any(word in query_lower for word in topic.split()):
            results.append(f"{topic.title()}: {info}")
    
    if results:
        return "\n".join(results)
    else:
        return "No relevant information found in the knowledge base. Try rephrasing your query or use web_search for external information."


# =============================================================================
# REASONING/LOGIC TOOL
# =============================================================================

class ReasoningInput(BaseModel):
    problem: str = Field(description="Logic or reasoning problem to solve")


@tool("logical_reasoning", args_schema=ReasoningInput)
def logical_reasoning(problem: str) -> str:
    """
    Solve logical reasoning problems step by step.
    Use this for puzzles, deductions, and analytical problems.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a logical reasoning expert. Solve problems step by step.
For each step:
1. State the given information
2. Identify what needs to be found
3. Apply logical rules
4. Show your reasoning
5. State the conclusion"""),
        ("human", "{problem}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"problem": problem})
        return result
    except Exception as e:
        return f"Reasoning error: {str(e)}"


# =============================================================================
# TOOL REGISTRY
# =============================================================================

def get_all_tools() -> List:
    """Get all available tools"""
    return [
        web_search,
        calculator,
        python_executor,
        news_search,
        sql_query,
        datetime_tool,
        knowledge_base,
        logical_reasoning,
        wolfram_alpha,
    ]


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all tools"""
    tools = get_all_tools()
    descriptions = []
    
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    
    return "\n".join(descriptions)
