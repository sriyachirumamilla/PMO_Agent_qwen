"""
PMO Agent Qwen - Interactive Mode (FIXED IMPORTS)
"""
import os
import sys

# FIX: Add CURRENT DIRECTORY to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_qwen import create_pmo_agent_qwen, test_qwen_agent
    from langchain_core.messages import HumanMessage
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("📁 Make sure agent_qwen.py is in the same folder as run.py")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 PMO Agent Qwen - Interactive Mode")
    print("=" * 60)
    
    # Quick demo
    test_qwen_agent()
    
    # INTERACTIVE CHAT
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE")
    print("📝 Examples: 'Create task UI design', 'List all tasks'")
    print("💡 Type 'quit' to exit")
    print("="*60)
    
    agent = create_pmo_agent_qwen()
    config = {"configurable": {"thread_id": "chat"}}
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config
            )
            print(f"\n🤖 Agent: {response['messages'][-1].content}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")