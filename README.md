# 🤖 Multi-Tool MRKL Agent

A sophisticated reasoning agent based on the **MRKL (Modular Reasoning, Knowledge and Language)** architecture. This agent dynamically selects and chains tools to solve complex, multi-step problems.

## 🎯 Key Features

- **Dynamic Tool Selection**: Automatically chooses the right tool for each subtask
- **Chain-of-Thought Reasoning**: Explicit reasoning steps for transparency
- **Multi-Step Problem Solving**: Breaks complex tasks into manageable steps
- **Self-Correction**: Handles tool failures gracefully
- **Planning Mode**: Optional explicit planning before execution
- **Streaming Output**: Real-time visibility into agent reasoning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
│           "What's the weather in Tokyo and convert              │
│            the temperature to Fahrenheit?"                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Planning Phase (Optional)                     │
│  1. Search for Tokyo weather                                     │
│  2. Extract temperature value                                    │
│  3. Convert Celsius to Fahrenheit using calculator              │
│  4. Format final answer                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reasoning Loop                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Thought: I need to find the current weather in Tokyo     │ │
│  │  Action: web_search                                        │ │
│  │  Action Input: "Tokyo weather today"                       │ │
│  │  Observation: Temperature: 22°C, Partly cloudy            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Thought: Now I need to convert 22°C to Fahrenheit        │ │
│  │  Action: calculator                                        │ │
│  │  Action Input: "22 * 9/5 + 32"                            │ │
│  │  Observation: Result: 71.6                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Thought: I have all the information needed               │ │
│  │  Final Answer: Tokyo is currently 22°C (71.6°F)           │ │
│  │               with partly cloudy conditions.               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `web_search` | Search the internet | Current information, facts |
| `calculator` | Math operations | Calculations, conversions |
| `python_executor` | Run Python code | Data processing, algorithms |
| `wolfram_alpha` | Scientific computing | Complex math, science data |
| `news_search` | Recent news | Current events |
| `sql_query` | Database queries | Structured data |
| `datetime_tool` | Date/time operations | Scheduling, time zones |
| `knowledge_base` | Internal docs | Company-specific info |
| `logical_reasoning` | Logic problems | Puzzles, deductions |

## 📁 Project Structure

```
mrkl-agent/
├── config.py           # Configuration
├── tools.py            # Tool implementations
├── agent.py            # MRKL agent core
├── main.py             # CLI and API
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=sk-...
export SERPAPI_KEY=...  # Optional, for web search
```

### Usage

**Interactive CLI:**
```bash
python main.py

# Example queries:
# "What's 25% of the current Bitcoin price?"
# "How many days until New Year and what day will it be?"
# "Search for recent AI news and summarize the top story"
```

**Single Query:**
```bash
python main.py --query "Calculate the compound interest on $10,000 at 5% for 10 years"
```

**With Planning:**
```bash
python main.py --query "Compare the populations of Tokyo and New York" --plan
```

**API Server:**
```bash
python main.py --mode api --port 8000
```

### API Usage

```bash
# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 15% tip on $84.50?", "with_planning": false}'

# Stream execution
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Find the latest stock price of Apple"}'

# Generate plan only
curl -X POST "http://localhost:8000/plan" \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan a trip from NYC to LA"}'
```

### Python API

```python
from agent import MRKLAgent, ask_agent

# Simple usage
result = ask_agent("What's the capital of France and its population?")
print(result["answer"])

# With planning
result = ask_agent("Calculate monthly payments for a $300k mortgage at 7% for 30 years", with_planning=True)
print("Plan:", result["plan"])
print("Answer:", result["answer"])

# Streaming
agent = MRKLAgent()
for event in agent.stream("How old is the universe in seconds?"):
    print(event)
```

## 🎮 Example Queries

### Math & Calculations
- "What's 15% of 847.50?"
- "Convert 100 kilometers to miles"
- "Calculate compound interest on $5000 at 4% for 5 years"

### Current Information
- "What's the current price of Bitcoin?"
- "Latest news about artificial intelligence"
- "Weather in London today"

### Multi-Step Reasoning
- "How many seconds are there until New Year 2025?"
- "What's 20% tip on a $85 dinner bill, split 4 ways?"
- "Find Apple's stock price and calculate its market cap"

### Knowledge & Research
- "What is the refund policy?" (from knowledge base)
- "Compare Python and JavaScript for web development"
- "Explain quantum computing in simple terms"

## 🔄 Reasoning Flow

```
1. RECEIVE QUERY
   ↓
2. THINK: Analyze what's needed
   ↓
3. SELECT TOOL: Choose appropriate tool
   ↓
4. EXECUTE: Run tool with input
   ↓
5. OBSERVE: Process tool output
   ↓
6. ITERATE: Need more info? → Go to step 2
   ↓
7. ANSWER: Formulate final response
```

## 🛡️ Error Handling

- **Tool Failure**: Agent tries alternative approaches
- **Invalid Input**: Graceful error messages
- **Max Iterations**: Prevents infinite loops
- **Timeout Protection**: Limits execution time

## 🔧 Configuration

```python
# config.py
MAX_ITERATIONS = 10        # Max reasoning steps
MAX_EXECUTION_TIME = 120   # Timeout in seconds
VERBOSE = True             # Show reasoning steps
TEMPERATURE = 0.1          # LLM temperature (low for consistency)
```

## 🎯 Design Principles

1. **Modularity**: Each tool is independent and replaceable
2. **Transparency**: All reasoning steps are visible
3. **Robustness**: Handles failures gracefully
4. **Extensibility**: Easy to add new tools
5. **Efficiency**: Minimal tool calls to solve tasks

## 📊 Performance Tips

- Use specific queries for better tool selection
- Enable planning for complex multi-step tasks
- Provide context when needed
- Check tool availability for your use case

## 🤝 Adding Custom Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param: str = Field(description="Parameter description")

@tool("my_tool", args_schema=MyToolInput)
def my_tool(param: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result

# Add to tools.py get_all_tools()
```

## 📝 License

MIT License

---

