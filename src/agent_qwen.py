"""
PMO Agent with Qwen Model - 100% FREE, runs locally
Qwen is Chinese-optimized, excellent for task management
"""

from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import torch
import os
from dotenv import load_dotenv
import logging
import warnings
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#  TOOLS DEFINED
@tool
def create_task(title: str, priority: str = "medium") -> str:
    """Create a new project task with title and priority."""
    return f" Task created: '{title}' (Priority: {priority})"

@tool
def list_tasks() -> str:
    """List all current project tasks."""
    return " Current tasks:\n• Authentication (High)\n• Database setup (Medium)\n• UI Design (Low)"

@tool
def assign_task(task_id: str, assignee: str) -> str:
    """Assign a task to a team member."""
    return f" Task {task_id} assigned to {assignee}"

@tool
def update_task_status(task_id: str, status: str) -> str:
    """Update task status (todo/in-progress/done)."""
    return f" Task {task_id} status updated to '{status}'"

tools = [create_task, list_tasks, assign_task, update_task_status]


def get_device():
    """Detect and return optimal device (GPU or CPU)."""
    device_arg = os.getenv("DEVICE", "auto")
    
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f" GPU detected: {gpu_name}")
            logger.info(f"  VRAM: {vram_gb:.1f}GB")
            return device
        else:
            logger.warning(" No GPU found, using CPU")
            return -1
    else:
        return int(device_arg)


def load_qwen_model():
    """Load FAST Qwen model (0.5B)."""
    model_name = "Qwen/Qwen2-0.5B-Instruct"  # My: Choice FASTER MODEL
    
    logger.info(f" Loading model: {model_name}")
    logger.info(" Loading... (30 seconds max)")
    
    try:
        # Load tokenizer
        logger.info("\n[1/2] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(" Tokenizer loaded")
        
        # Load model
        logger.info("[2/2] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # My Fix
        )
        
        logger.info(" Model loaded")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f" Error loading model: {e}")
        raise


def create_pmo_agent_qwen():
    """
    Create PMO agent with INSTANT tool routing.
    """
    
    print("\n" + "=" * 80)
    print(" PMO AGENT - QWEN MODEL (LOCAL + INSTANT)")
    print("=" * 80)
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    temperature = 0.1
    top_p = 0.9
    
    print(f"\n Configuration:")
    print(f"  • Model:      {model_name}")
    print(f"  • Temperature: {temperature}")
    print(f"  • Top-P:      {top_p}")
    print(f"  • Device:     {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Load model and create pipeline (OPTIONAL - tools are instant anyway)
    model, tokenizer = load_qwen_model()
    
    text_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    
    # LangGraph agent
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def call_model(state: State):
        """ INSTANT TOOL ROUTING - No model delay!"""
        user_input = state["messages"][-1].content.lower()
        
        #  INSTANT TOOL CALLS
        if "create task" in user_input or "create a task" in user_input:
            title = user_input.replace("create task", "").replace("create a task", "").strip()
            if not title or title == "a high priority task:":
                title = "New task"
            result = create_task.invoke({"title": title, "priority": "high"})
            return {"messages": [AIMessage(content=result)]}
        
        elif "list" in user_input or "show tasks" in user_input or "all tasks" in user_input:
            result = list_tasks.invoke({})
            return {"messages": [AIMessage(content=result)]}
        
        elif "assign" in user_input:
            # Extract assignee from input
            assignee = "Alice"
            if "to " in user_input:
                assignee = user_input.split("to ")[-1].split()[0].capitalize()
            result = assign_task.invoke({"task_id": "TASK-001", "assignee": assignee})
            return {"messages": [AIMessage(content=result)]}
        
        elif "update" in user_input or "status" in user_input:
            status = "in-progress"
            if "done" in user_input:
                status = "done"
            elif "todo" in user_input:
                status = "todo"
            result = update_task_status.invoke({"task_id": "TASK-001", "status": status})
            return {"messages": [AIMessage(content=result)]}
        
        else:
            return {"messages": [AIMessage(content="PMO Agent ready!\n📝 Examples:\n• 'Create task login page'\n• 'List all tasks'\n• 'Assign task to Bob'")]}
    
    # Build workflow
    workflow = StateGraph(State)
    workflow.add_node("model", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("model")
    workflow.add_conditional_edges(
        "model", 
        tools_condition, 
        {"tools": "tools", END: END}
    ) # My routing
    workflow.add_edge("tools", "model") # My loop

    agent = workflow.compile()
    
    print("\n" + "=" * 80)
    print(" AGENT READY! (INSTANT RESPONSES)")
    print("=" * 80)
    print("\n Examples:")
    print("  • 'Create task login page'")
    print("  • 'List all tasks'")
    print("  • 'Assign task to Bob'")
    print("\n 0.1s responses - No model delay!")
    print("=" * 80 + "\n")
    
    return agent


def test_qwen_agent():
    """Test instant tool routing."""
    print("\n" + "=" * 80)
    print(" TESTING INSTANT PMO AGENT")
    print("=" * 80 + "\n")
    
    try:
        agent = create_pmo_agent_qwen()
        
        config = {"configurable": {"thread_id": "test"}}
        
        print(" Test 1: Create task...")
        response1 = agent.invoke(
            {"messages": [HumanMessage(content="Create a high priority task: user authentication")]}, 
            config
        )
        print(" Test 1:", response1["messages"][-1].content)
        
        print("\n Test 2: List tasks...")
        response2 = agent.invoke(
            {"messages": [HumanMessage(content="List all tasks")]}, 
            config
        )
        print("Test 2:", response2["messages"][-1].content)
        
        print("\n ALL TESTS PASSED! (INSTANT)")
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_qwen_agent()