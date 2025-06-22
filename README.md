# LangGraph Supervisor with Memory Trimming (Gemini Edition)

This repository contains implementations of LangGraph supervisor agents with memory trimming functionality that keeps only the last K messages to prevent context overflow. **Now powered by Google's Gemini 2.5 Flash Preview model!**

## Features

- **Memory Trimming**: Automatically keeps only the last K messages in conversation history
- **Supervisor Architecture**: Central supervisor coordinates multiple specialized agents
- **Multi-Agent System**: Specialized agents for math, research, and writing tasks
- **Persistent Memory**: Uses LangGraph's MemorySaver for conversation persistence
- **Flexible Configuration**: Configurable memory limits and agent capabilities
- **Gemini Integration**: Powered by Google's latest Gemini 2.5 Flash Preview model

## Files

### 1. `simple_memory_trimming_gemini.py`
Basic example of memory trimming with a single agent using Gemini, closely following the provided example code.

### 2. `supervisor_with_memory_gemini.py`
Comprehensive supervisor implementation with:
- Math expert agent (arithmetic, percentages)
- Research expert agent (web search, fact finding)
- Writing expert agent (reports, summaries)
- Memory trimming functionality
- Detailed conversation management
- Gemini 2.5 Flash Preview integration

### 3. `advanced_supervisor_memory_gemini.py`
Advanced implementation with:
- Custom memory trimming logic integrated into supervisor
- Real-time memory status monitoring
- Sophisticated agent routing
- Memory usage analytics
- Optimized for Gemini model performance

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv langgraph_env

# Activate virtual environment
# On Windows:
langgraph_env\Scripts\activate
# On macOS/Linux:
source langgraph_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
# Copy the template
copy .env.template .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_google_api_key_here
```

**Note**: You'll need a Google API key with access to Gemini models. Get yours at [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

### Simple Memory Trimming Example

```bash
python simple_memory_trimming_gemini.py
```

This demonstrates basic memory trimming with a single agent that:
- Keeps only the last 5 messages
- Shows memory status during conversation
- Tests memory limits with extended conversations
- Uses Gemini 2.5 Flash Preview model

### Comprehensive Supervisor Example

```bash
python supervisor_with_memory_gemini.py
```

This runs a full supervisor system that:
- Manages 3 specialized agents
- Applies memory trimming (configurable limit)
- Shows real-time memory usage
- Demonstrates multi-agent coordination
- Powered by Gemini 2.5 Flash Preview

### Advanced Supervisor Example

```bash
python advanced_supervisor_memory_gemini.py
```

This showcases advanced features:
- Custom memory trimming logic
- Memory analytics
- Sophisticated routing
- Real-time memory monitoring
- Optimized Gemini integration

## Memory Trimming Configuration

The memory trimming uses LangChain's `trim_messages` function with these parameters:

```python
selected_messages = trim_messages(
    messages,
    token_counter=len,        # Count messages, not tokens
    max_tokens=max_messages,  # Maximum number of messages
    strategy="last",          # Keep the last N messages
    start_on="human",         # Ensure valid chat history
    include_system=True,      # Keep system messages
    allow_partial=False,      # Don't allow partial messages
)
```

## Architecture

```
┌─────────────────┐
│   Supervisor    │
│   (with memory  │
│    trimming)    │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐   ┌─────────┐
│ Math  │   │Research│   │ Writing │
│Expert │   │ Expert │   │ Expert  │
└───────┘   └───────┘   └─────────┘
```

## Memory Management

- **Memory Limit**: Configurable (default: 5-10 messages)
- **Trimming Strategy**: Keep last K messages
- **System Messages**: Always preserved
- **Chat History**: Maintains conversation flow
- **Thread Management**: Each conversation has unique thread ID

## Example Output

```
🤖 LangGraph Supervisor with Memory Trimming (Gemini 2.5 Flash)
===============================================================
Memory limit: 5 messages
Powered by: Gemini 2.5 Flash Preview

--- Query 1 ---
👤 User: What is 15% of 240?
🧠 Memory Status: Trimmed from 2 to 2 messages
🤖 Assistant: 15% of 240 is 36.

--- Query 6 ---
👤 User: Do you remember the first calculation?
🧠 Memory Status: Trimmed from 12 to 5 messages
🤖 Assistant: I can see recent calculations, but the first one may have been trimmed from memory due to the conversation length.
```

## Gemini Model Configuration

The implementations use Google's Gemini 2.5 Flash Preview model with these settings:

```python
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

## Customization

### Adjust Memory Limit

```python
supervisor = SupervisorWithMemoryTrimming(max_messages=10)  # Keep 10 messages
```

### Add Custom Agents

```python
custom_agent = create_react_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20"),
    tools=[your_custom_tools],
    name="custom_expert",
    prompt="Your custom prompt"
)
```

### Modify Model Settings

```python
# Adjust Gemini model parameters
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.5,  # Lower for more deterministic responses
    max_tokens=2048,  # Set token limit if needed
    max_retries=3     # Increase retries for reliability
)
```

### Modify Trimming Strategy

```python
# Use token-based trimming instead of message count
# Note: For Gemini, you can use the model itself as token counter
selected_messages = trim_messages(
    messages,
    token_counter=model,  # Use Gemini model's token counter
    max_tokens=1000,     # Token limit
    strategy="last"
)
```

## Requirements

- Python 3.10+
- Google API key with Gemini access
- See `requirements.txt` for full dependencies

## Getting Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file as `GOOGLE_API_KEY`

## Model Information

**Gemini 2.5 Flash Preview** features:
- Fast inference speed
- High-quality responses
- Multimodal capabilities
- Cost-effective pricing
- Latest model architecture from Google

## License

MIT License
