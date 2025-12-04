# 🚀 PMO Task Agent 

**PMO Task Management Agent built with LangGraph + Qwen2**

[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-blue)](https://langchain.com)
[![Qwen2](https://img.shields.io/badge/Model-Qwen2--0.5B-green)](https://huggingface.co/Qwen)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://python.org)

You: Create task login page
Agent: ✅ Task created: 'login page' (Priority: high)

You: List all tasks
Agent: 📋 Current tasks:
• Authentication (High)
• login page (High)
• Database setup (Medium)

✨ Features

✅ 100% Local — No API keys or costs (Qwen2-0.5B CPU)

⚡ Instant Tools — ~0.1s responses (custom routing)

🧠 LangGraph State — Full conversation memory

🧰 4 Built-in PMO Tools

1.Create Task
2.List Tasks
3.Assign Task
4.Update Task Status

## 🏗️ Architecture
User Input
   ↓
LangGraph Router
   ↓
PMO Tools (create, list, assign, update)
   ↓
Stateful Response

Example:
"Create task X" → create_task() → Tasks updated → Returns result



## 🚀 Quick Start

# Clone & Install
git clone https://github.com/YOUR_USERNAME/pmo-task-agent.git
cd pmo-task-agent/src
pip install -r requirements.txt

# Run Agent
python run.py



Try these:

Create task sprint planning
List all tasks
Assign task to axe
quit

📦 File Structure

pmo-task-agent/
├── src/
│   ├── run.py             # Interactive launcher
│   ├── agent_qwen.py      # The PMO Agent (LangGraph + Qwen)
│   ├── requirements.txt
│   └── README.md

📝 Requirements

1.Python 3.10+
2.CPU only (Qwen2-0.5B) OR GPU (optional)
3.pip dependencies in requirements.txt

⭐️ Support

If you like this project, star ⭐ it on GitHub!