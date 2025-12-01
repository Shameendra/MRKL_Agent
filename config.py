"""
Configuration for Multi-Tool MRKL Agent
MRKL: Modular Reasoning, Knowledge and Language
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# LLM Settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Agent Settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "120"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# Tool Settings
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "30"))
