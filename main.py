"""
MRKL Agent Application
CLI and API interfaces
"""

import argparse
import json
from typing import Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from agent import MRKLAgent, MRKLAgentWithPlanning, ask_agent
from tools import get_all_tools, get_tool_descriptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MRKL Agent API",
    description="Multi-Tool Reasoning Agent with modular capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question or task for the agent")
    with_planning: bool = Field(default=False, description="Use planning phase")
    stream: bool = Field(default=False, description="Stream responses")


class QueryResponse(BaseModel):
    input: str
    answer: Optional[str]
    error: Optional[str]
    steps: list
    iterations: int
    plan: Optional[list] = None


class ToolInfo(BaseModel):
    name: str
    description: str


# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "MRKL Agent",
        "version": "1.0.0",
        "tools": len(get_all_tools())
    }


@app.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """List all available tools"""
    tools = get_all_tools()
    return [
        ToolInfo(name=tool.name, description=tool.description)
        for tool in tools
    ]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the MRKL agent"""
    try:
        result = ask_agent(
            request.question,
            with_planning=request.with_planning
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream agent execution"""
    
    async def generate():
        agent = MRKLAgent()
        
        for event in agent.stream(request.question):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/plan")
async def create_plan(request: QueryRequest):
    """Generate an execution plan without running"""
    agent = MRKLAgentWithPlanning()
    plan = agent.plan(request.question)
    
    return {
        "task": request.question,
        "plan": plan
    }


# CLI Interface
def interactive_cli():
    """Interactive command-line interface"""
    print("\n" + "=" * 60)
    print("🤖 MRKL Agent - Multi-Tool Reasoning")
    print("=" * 60)
    print("\nAvailable Tools:")
    for tool in get_all_tools():
        print(f"  • {tool.name}: {tool.description[:60]}...")
    print("\nCommands:")
    print("  /plan    - Show execution plan before running")
    print("  /tools   - List all tools")
    print("  /quit    - Exit")
    print("\nAsk me anything!\n")
    
    agent = MRKLAgentWithPlanning()
    use_planning = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "/tools":
                print("\nAvailable Tools:")
                for tool in get_all_tools():
                    print(f"  • {tool.name}")
                    print(f"    {tool.description}\n")
                continue
            
            if user_input.lower() == "/plan":
                use_planning = not use_planning
                status = "ON" if use_planning else "OFF"
                print(f"\n📋 Planning mode: {status}\n")
                continue
            
            print("\n🔍 Thinking...\n")
            
            if use_planning:
                # Show plan first
                plan = agent.plan(user_input)
                print("📋 Execution Plan:")
                for step in plan:
                    print(f"   {step}")
                print()
            
            # Stream the execution
            for event in agent.stream(user_input):
                if event["type"] == "thought":
                    print(f"💭 Thought: {event['thought']}")
                elif event["type"] == "tool_call":
                    print(f"🔧 Using: {event['tool']}")
                    print(f"   Input: {event['input'][:100]}...")
                elif event["type"] == "observation":
                    print(f"👁️ Result: {event['observation'][:200]}...")
                elif event["type"] == "final_answer":
                    print(f"\n✅ Answer: {event['answer']}")
                elif event["type"] == "error":
                    print(f"\n❌ Error: {event['message']}")
                print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MRKL Agent")
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "api"],
        default="cli",
        help="Run mode"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="API server port"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode"
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Use planning mode"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        result = ask_agent(args.query, with_planning=args.plan)
        
        print("\n" + "=" * 60)
        print("🤖 MRKL Agent Result")
        print("=" * 60)
        
        if result.get("plan"):
            print("\n📋 Plan:")
            for step in result["plan"]:
                print(f"   {step}")
        
        print(f"\n📝 Question: {result['input']}")
        
        if result.get("steps"):
            print("\n🔄 Reasoning Steps:")
            for i, step in enumerate(result["steps"], 1):
                print(f"\n  Step {i}:")
                print(f"    Thought: {step['thought'][:100]}...")
                if step.get("action"):
                    print(f"    Action: {step['action']}")
                if step.get("observation"):
                    print(f"    Observation: {step['observation'][:100]}...")
        
        if result.get("answer"):
            print(f"\n✅ Answer: {result['answer']}")
        
        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")
        
        print(f"\n📊 Iterations: {result['iterations']}")
        
    elif args.mode == "api":
        print(f"Starting MRKL Agent API on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()
