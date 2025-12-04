"""
Main interactive interface for PMO Agent.
Run this file to start the agent in chat mode.

Usage:
    python src/main.py
"""

from src.agent_qwen import create_pmo_agent_qwen
from src.models import TASKS_DB
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🚀 PMO TASK MANAGEMENT AGENT - POWERED BY QWEN       ║
║                                                              ║
║  100% FREE • OPEN SOURCE • RUNS LOCALLY • NO API KEYS       ║
║                                                              ║
║  Type 'help' for commands • 'quit' to exit                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
📖 **AVAILABLE COMMANDS:**

Task Management:
  • "create a task to [description]" - Create new task
  • "assign [TASK-ID] to [person]" - Assign task to someone
  • "mark [TASK-ID] as [status]" - Update task status
  • "show all tasks" - List all tasks
  • "show details of [TASK-ID]" - Get task details
  • "show statistics" - Get project stats

Examples:
  ✓ Create a task to implement user authentication with high priority
  ✓ Assign TASK-abc1 to Alice
  ✓ Mark TASK-abc1 as in_progress
  ✓ Show all tasks
  ✓ Show statistics

Other:
  • "help" - Show this help message
  • "quit" / "exit" / "q" - Exit the program
  • "clear" - Clear screen

💡 TIP: Be natural! The AI understands English and will figure out what you mean.
"""
    print(help_text)


def print_stats():
    """Print current task statistics."""
    if not TASKS_DB:
        print("📊 No tasks yet. Create one to see statistics.\n")
        return
    
    tasks = list(TASKS_DB.values())
    
    by_status = {}
    for task in tasks:
        status = task.status
        by_status[status] = by_status.get(status, 0) + 1
    
    print("\n📊 **QUICK STATS**")
    print(f"Total Tasks: {len(tasks)}")
    for status, count in by_status.items():
        print(f"  • {status}: {count}")
    print()


def main():
    """Main interactive loop."""
    print_banner()
    
    print("⏳ Loading Qwen model (first time: ~2-3 minutes)...")
    try:
        agent = create_pmo_agent_qwen()
        print("✓ Agent ready!\n")
    except Exception as e:
        print(f"✗ Error loading agent: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Ensure you have 8GB+ RAM")
        print("  2. First run downloads model (~3GB)")
        print("  3. Check internet connection")
        print("  4. Try: pip install -r requirements.txt\n")
        return
    
    print_help()
    print("=" * 60)
    print("Ready! Type your request below:\n")
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye! Your tasks have been saved in memory.")
                break
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            if user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue
            
            if user_input.lower() == "stats":
                print_stats()
                continue
            
            # Show thinking indicator
            print("\n🤔 Agent thinking", end="", flush=True)
            for _ in range(3):
                print(".", end="", flush=True)
                import time
                time.sleep(0.3)
            print(" \n")
            
            # Invoke agent
            try:
                response = agent.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                })
                
                # Extract final message
                final_message = response["messages"][-1]
                agent_response = final_message.content
                
                # Print response
                print(f"Agent: {agent_response}\n")
                
                # Track conversation
                conversation_history.append({
                    "user": user_input,
                    "agent": agent_response
                })
                
            except Exception as e:
                print(f"⚠️ Agent Error: {str(e)[:100]}")
                print("Try rephrasing your request.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()