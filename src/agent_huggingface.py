
from langchain.agents import create_agent
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import torch
import os
from dotenv import load_dotenv
from src.tools import tools
import logging

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_qwen_direct():
    
    model_name = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
    device = 0 if torch.cuda.is_available() else -1
    
    logger.info(f"📥 Loading {model_name}")
    logger.info("⏳ Loading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✓ Tokenizer loaded")
        logger.info("📥 Loading model...")
        
        if device >= 0:
            if torch.cuda.is_available():
                logger.info(f"🚀 GPU mode (float16)")
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                logger.info("⚠️  GPU requested but not available, falling back to CPU")
                device = -1
        
        if device >= 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            logger.info("💻 CPU mode (float32)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        
        logger.info("✓ Model loaded")
        return model, tokenizer, device
    
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        logger.info("\n🔧 TROUBLESHOOTING:")
        logger.info("  1. Check internet connection")
        logger.info("  2. Ensure 5GB+ disk space")
        logger.info("  3. Try: pip install -r requirements.txt")
        raise


def create_pmo_agent_huggingface():
    print("\n" + "=" * 80)
    print("🚀 PMO AGENT - QWEN (HuggingFace Pipeline)")
    print("=" * 80)
    
    max_length = int(os.getenv("MAX_LENGTH", "256"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    top_p = float(os.getenv("TOP_P", "0.9"))
    
    print(f"\n📋 Configuration:")
    print(f"  • Max Tokens: {max_length}")
    print(f"  • Temperature: {temperature}")
    print(f"  • Top-P:      {top_p}")
    print()
    
    try:
        # Load model
        model, tokenizer, device = load_qwen_direct()
        
        # Create pipeline
        logger.info("\n🔧 Creating text generation pipeline...")
        text_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            device=device,
        )
        logger.info("✓ Pipeline created")
        
        # Wrap with LangChain's HuggingFacePipeline
        logger.info("🔗 Wrapping with LangChain...")
        llm = HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={
                "temperature": temperature,
                "max_length": max_length,
            },
        )
        logger.info("✓ LLM wrapper ready")
        
        # Create agent
        logger.info("⚙️  Creating agent with tools...")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""You are a helpful PMO (Project Management Office) Assistant.
Your job is to help manage project tasks, assign work, and track progress.

When users ask you to do something, use the available tools to:
1. Create new tasks with clear descriptions and priorities
2. Assign tasks to team members  
3. Update task status as work progresses
4. Show project information and statistics
5. Provide project insights and metrics

Guidelines:
- Be professional, clear, and helpful in all responses
- Ask for clarification if user requests are ambiguous
- Provide confirmation when actions are completed
- Use task IDs when referencing specific tasks
- Support both English and Chinese requests
- Help users stay organized and on track"""
        )
        
        logger.info("✓ Agent created successfully")
        
        print("\n" + "=" * 80)
        print("✅ AGENT READY!")
        print("=" * 80)
        print("\n📝 Examples:")
        print("  • 'Create a task to implement authentication'")
        print("  • 'Assign TASK-xxx to Alice'")
        print("  • 'Mark TASK-xxx as in_progress'")
        print("  • 'Show all tasks'")
        print("  • 'Show statistics'")
        print("\n" + "=" * 80 + "\n")
        
        return agent
    
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise


def create_lightweight_huggingface_agent():
    print("🚀 PMO AGENT (LIGHTWEIGHT - QWEN 0.5B - HuggingFace)")
    print("=" * 80)
    
    os.environ["QWEN_MODEL_NAME"] = "Qwen/Qwen2-0.5B-Instruct"
    os.environ["MAX_LENGTH"] = "128"
    os.environ["TEMPERATURE"] = "0.5"
    
    return create_pmo_agent_huggingface()


def create_high_quality_huggingface_agent():
    print("🚀 PMO AGENT (HIGH-QUALITY - QWEN 7B - HuggingFace)")
    print("=" * 80)
    print("⚠️  Requires: GPU with 16GB+ VRAM")
    print()
    
    os.environ["QWEN_MODEL_NAME"] = "Qwen/Qwen2-7B-Instruct"
    os.environ["MAX_LENGTH"] = "512"
    os.environ["TEMPERATURE"] = "0.7"
    
    return create_pmo_agent_huggingface()


def test_huggingface_agent():
    """Test if HuggingFace agent works correctly."""
    print("\n" + "=" * 80)
    print("🧪 TESTING HUGGINGFACE AGENT")
    print("=" * 80 + "\n")
    
    try:
        agent = create_pmo_agent_huggingface()
        
        print("📝 Test 1: Creating a task...")
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Create a task to implement authentication with high priority"
            }]
        })
        print("✓ Response:")
        print(response["messages"][-1].content)
        print()
        
        print("📝 Test 2: Listing tasks...")
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Show all tasks"
            }]
        })
        print("✓ Response:")
        print(response["messages"][-1].content)
        print()
        
        print("📝 Test 3: Getting statistics...")
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Show statistics"
            }]
        })
        print("✓ Response:")
        print(response["messages"][-1].content)
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_huggingface_agent()