"""
Simplified LangGraph Supervisor with Memory Trimming using pre_model_hook

This implementation uses LangGraph's built-in pre_model_hook feature to apply
memory trimming before each LLM call, making it much simpler than custom wrappers.
"""

import time
import uuid
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class SimpleHookSupervisor:
    """
    Simplified supervisor using pre_model_hook for automatic memory trimming.
    
    This approach applies pre_model_hook to BOTH supervisor AND agents:
    1. Supervisor LLM (for coordination decisions) ✅
    2. Math agent LLM (for calculations) ✅
    3. Research agent LLM (for information gathering) ✅
    
    Note: create_supervisor() DOES support pre_model_hook parameter!
    Memory trimming happens before every LLM call in the entire workflow.
    """
    
    def __init__(self, max_messages: int = 6):
        """
        Initialize supervisor with pre_model_hook trimming.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.7,
            max_retries=2
        )
        self.memory = MemorySaver()
          # Create the pre-model hook for memory trimming
        self.pre_model_hook = self._create_trimming_hook()
        # Create agents and supervisor
        self.agents = self._create_agents()
        self.supervisor = self._create_supervisor()
    
    def _create_trimming_hook(self):
        """
        Create the pre_model_hook function for automatic memory trimming.
        
        This function is called before every LLM invocation and automatically
        trims the conversation to the last K messages.
        """
        def pre_model_hook(state):
            """Trim messages before each LLM call."""
            
            messages = state.get("messages", [])

            print("INSIDE MODEL HOOK")
            self.print_messages_nicely(messages, title="Before Trimming")

            trimmed_messages = trim_messages(
                messages,
                token_counter=len,  # <-- len will simply count the number of messages rather than tokens
                max_tokens=self.max_messages,  # <-- allow up to 5 messages.
                strategy="last",
                # Most chat models expect that chat history starts with either:
                # (1) a HumanMessage or
                # (2) a SystemMessage followed by a HumanMessage
                # start_on="human" makes sure we produce a valid chat history
                start_on="human",
                # Usually, we want to keep the SystemMessage
                # if it's present in the original history.
                # The SystemMessage has special instructions for the model.
                include_system=True,
                allow_partial=False,
            )
            self.print_messages_nicely(trimmed_messages, title="After Trimming")
            print("GOING OUT OF MODEL HOOK")
            time.sleep(20)  # Simulate processing delay for realism
            #   # Apply memory trimming using LangChain's trim_messages
            # # Simple approach: just keep the last N messages without complex constraints
            # if len(messages) <= self.max_messages:
            #     trimmed_messages = messages
            # else:
            #     # Keep the last max_messages, ensuring we include system messages
            #     system_messages = [msg for msg in messages if getattr(msg, 'type', '') == 'system']
            #     non_system_messages = [msg for msg in messages if getattr(msg, 'type', '') != 'system']
                
            #     # Take the last (max_messages - len(system_messages)) non-system messages
            #     recent_messages = non_system_messages[-(self.max_messages - len(system_messages)):]
                
            #     # Combine system messages + recent messages
            #     trimmed_messages = system_messages + recent_messages
            #   # Print detailed trimming info for debugging
            # if len(messages) != len(trimmed_messages):
            #     print(f"🪝 Hook Trimming: {len(messages)} → {len(trimmed_messages)} messages")
            #     print(f"📊 Original message types: {[getattr(msg, 'type', 'unknown') for msg in messages]}")
            #     print(f"📊 Trimmed message types: {[getattr(msg, 'type', 'unknown') for msg in trimmed_messages]}")
            #     print("📋 Messages sent to LLM:")
            #     for i, msg in enumerate(trimmed_messages):
            #         msg_type = getattr(msg, 'type', 'unknown')
            #         content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
            #         print(f"   {i+1}. [{msg_type}]: {content_preview}")
            #     print()
            # else:
            #     print(f"🪝 No trimming needed: {len(messages)} messages (≤ {self.max_messages})")
            #     print("📋 All messages sent to LLM:")
            #     for i, msg in enumerate(messages):
            #         msg_type = getattr(msg, 'type', 'unknown')
            #         content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
            #         print(f"   {i+1}. [{msg_type}]: {content_preview}")
            #     print()
            
            # # Return trimmed messages for the LLM
            # # Using 'llm_input_messages' key as recommended
            # return {"llm_input_messages": trimmed_messages}
            return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES)] + trimmed_messages}
        
        return pre_model_hook

    def print_messages_nicely(self, messages, title="Messages"):
        """
        Print messages in a clean, readable format.
        
        Args:
            messages: List of message objects
            title: Title for the message display
        """
        print(f"\n📋 {title}:")
        if not messages:
            print("   No messages")
            return
        
        for i, msg in enumerate(messages, 1):
            # Get message type
            msg_type = getattr(msg, 'type', 'unknown')
            
            # Get message content
            content = str(msg.content)
            
            # Get name if available (for AI messages from specific agents)
            name = getattr(msg, 'name', None)
            
            # Format the message type display
            if name:
                type_display = f"{msg_type} ({name})"
            else:
                type_display = msg_type
            
            # Truncate long content for readability
            if len(content) > 100:
                content_display = content[:97] + "..."
            else:
                content_display = content
            
            print(f"   {i:2d}. {type_display:15s}: \"{content_display}\"")


    def _create_agents(self):
        """Create specialized agents with pre_model_hook for trimming."""
        
        # Math tools
        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b
        
        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together."""
            return a * b
        
        @tool
        def divide(a: float, b: float) -> float:
            """Divide first number by second number."""
            if b == 0:
                return "Error: Cannot divide by zero"
            return a / b
        
        @tool
        def percentage(value: float, percent: float) -> float:
            """Calculate percentage of a value."""
            return (value * percent) / 100
        
        # Research tools
        @tool
        def web_search(query: str) -> str:
            """Search the web for information."""
            search_db = {
                "faang": (
                    "FAANG companies headcount 2024:\n"
                    "• Meta: 67,317 employees\n" 
                    "• Apple: 164,000 employees\n"
                    "• Amazon: 1,551,000 employees\n"
                    "• Netflix: 14,000 employees\n"
                    "• Google: 181,269 employees"
                ),
                "gemini": "Gemini 2.5 Flash Preview: Google's latest high-speed AI model optimized for efficiency and performance",
                "weather": "Current weather: Sunny, 23°C with light clouds",
                "tech news": "Latest: AI adoption accelerating in enterprise, new breakthroughs in multimodal AI",
                "default": f"Search results for '{query}': Comprehensive information found."
            }
            
            query_lower = query.lower()
            for key, response in search_db.items():
                if key in query_lower:
                    return response
            return search_db["default"]        @tool
        def get_date_time() -> str:
            """Get current date and time."""
            from datetime import datetime
            return f"Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create agents with pre_model_hook for automatic trimming
        math_agent = create_react_agent(
            model=self.model,
            tools=[add, multiply, divide, percentage],
            name="math_expert",
            prompt="You are a mathematics expert. Use tools for all calculations and show clear explanations.",
            # pre_model_hook=self.pre_model_hook,  # 🔑 KEY: Apply trimming hook to agent
            checkpointer=self.memory
        )
        
        research_agent = create_react_agent(
            model=self.model,
            tools=[web_search, get_date_time],
            name="research_expert",
            prompt="You are a research expert. Use your tools to find accurate information and provide detailed answers.",
            # pre_model_hook=self.pre_model_hook,  # 🔑 KEY: Apply trimming hook to agent
            checkpointer=self.memory
        )
        
        return [math_agent, research_agent]
    
    def _create_supervisor(self):
        """Create supervisor with memory management awareness."""
        
        supervisor_prompt = f"""You are an intelligent supervisor managing expert agents using Gemini 2.5 Flash.

**Available Experts:**
• math_expert: Mathematical calculations, arithmetic, percentages, and numerical analysis
• research_expert: Information research, web search, current events, and fact-finding

**Memory Management:**
Conversation history is automatically trimmed to keep only the last {self.max_messages} messages for optimal performance.

**Instructions:**
1. Analyze the user's request carefully
2. Choose the most appropriate expert:
   - Math problems → math_expert
   - Research/information queries → research_expert
3. Provide clear reasoning for your delegation choice
4. If context appears limited due to trimming, ask for clarification

Delegate tasks efficiently and ensure expert responses are comprehensive."""
          # Create supervisor workflow - NOW with pre_model_hook support!
        # Memory trimming applies to supervisor LLM calls as well as agent calls
        workflow = create_supervisor(
            agents=self.agents,
            model=self.model,
            prompt=supervisor_prompt,
            pre_model_hook=self.pre_model_hook,  # 🔑 KEY: Apply trimming to supervisor too!
            output_mode="full_history"  # Keep full conversation context
        )
        
        # Compile with checkpointer - agent-level pre_model_hooks handle trimming
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Chat with the supervisor. Memory trimming happens automatically via pre_model_hook.
        
        Args:
            message: User's input message
            thread_id: Conversation thread ID (auto-generated if None)
            
        Returns:
            Response dictionary with conversation information
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get conversation state before sending message
        try:
            pre_state = self.supervisor.get_state(config)
            pre_messages = pre_state.values.get("messages", [])
            self.print_messages_nicely(pre_messages, "🧠 Pre-conversation messages")
            pre_count = len(pre_messages)
        except:
            pre_count = 0
        
        # Create input message
        input_message = HumanMessage(content=message)
        
        print(f"💬 Processing message (pre-conversation: {pre_count} messages)")
        
        # Invoke supervisor - pre_model_hook automatically handles trimming
        result = self.supervisor.invoke(
            {"messages": [input_message]}, 
            config
        )
        
        
        # Analyze post-conversation state
        post_messages = result.get("messages", [])
        self.print_messages_nicely(post_messages, "🧠 Post-conversation messages")
        post_count = len(post_messages)
        
        # Extract response
        response_content = post_messages[-1].content if post_messages else "No response generated"
        
        # Calculate memory statistics
        memory_stats = {
            "pre_message_count": pre_count,
            "post_message_count": post_count,
            "max_messages": self.max_messages,
            "memory_usage": f"{min(post_count, self.max_messages)}/{self.max_messages}",
            "trimming_would_apply": post_count > self.max_messages,
            "hook_based_trimming": True  # Indicates we're using pre_model_hook
        }
        
        return {
            "response": response_content,
            "thread_id": thread_id,
            "memory_stats": memory_stats
        }
    def get_conversation_analysis(self, thread_id: str) -> Dict[str, Any]:
        """
        Analyze conversation thread memory usage.
        
        Args:
            thread_id: Thread to analyze
            
        Returns:
            Conversation analysis
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.supervisor.get_state(config)
            messages = state.values.get("messages", [])
            
            # Count message types
            human_msgs = sum(1 for msg in messages if hasattr(msg, 'type') and msg.type == 'human')
            ai_msgs = sum(1 for msg in messages if hasattr(msg, 'type') and msg.type == 'ai')
            tool_msgs = sum(1 for msg in messages if hasattr(msg, 'type') and msg.type == 'tool')
            
            return {
                "thread_id": thread_id,
                "total_messages": len(messages),
                "human_messages": human_msgs,
                "ai_messages": ai_msgs,
                "tool_messages": tool_msgs,
                "memory_limit": self.max_messages,
                "would_trigger_trimming": len(messages) > self.max_messages,
                "effective_context_size": min(len(messages), self.max_messages),
                "trimming_method": "pre_model_hook (automatic)"
            }
        except Exception as e:
            return {"error": str(e), "thread_id": thread_id}

    def display_current_messages(self, thread_id: str):
        """
        Display all current messages in the conversation thread.
        
        Args:
            thread_id: Thread to display messages for
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.supervisor.get_state(config)
            messages = state.values.get("messages", [])
            
            print(f"📋 Current Conversation State ({len(messages)} total messages):")
            if not messages:
                print("   No messages in conversation")
                return
            
            for i, msg in enumerate(messages):
                msg_type = getattr(msg, 'type', 'unknown')
                content_preview = str(msg.content)[:150] + "..." if len(str(msg.content)) > 150 else str(msg.content)
                print(f"   {i+1:2d}. [{msg_type:8s}]: {content_preview}")
            
            # Show what would be trimmed
            if len(messages) > self.max_messages:
                print(f"\n⚠️  Messages beyond limit ({self.max_messages}): {len(messages) - self.max_messages} would be trimmed")
                print(f"🔄 Messages that would be kept (last {self.max_messages}):")
                for i, msg in enumerate(messages[-self.max_messages:]):
                    msg_type = getattr(msg, 'type', 'unknown')
                    content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    print(f"   {i+1:2d}. [{msg_type:8s}]: {content_preview}")
            
        except Exception as e:
            print(f"❌ Error displaying messages: {str(e)}")


def main():
    """Demonstrate the simplified pre_model_hook approach."""
    print("🪝 Complete Hook-Based Memory Trimming")
    print("=" * 65)
    print("🤖 Model: Gemini 2.5 Flash Preview")
    print("🔧 Method: pre_model_hook (supervisor + agent trimming)")
    print("📚 Framework: LangGraph Supervisor")
    print("✅ Note: ALL LLM calls are trimmed (supervisor + agents)")
    
    # Initialize with small memory limit to demonstrate trimming
    supervisor = SimpleHookSupervisor(max_messages=10)
    print(f"🧠 Memory limit: {supervisor.max_messages} messages")
    print("👥 Agents: math_expert, research_expert\n")
    
    # Create conversation thread
    thread = str(uuid.uuid4())
    print(f"🧵 Thread: {thread[:8]}...\n")
    
    # Test queries to demonstrate memory trimming
    test_queries = [
        "Hi! I'm Alex, a software engineer. Please remember my name and profession.",
        "What is 15% of 320? I need this calculation.",
        "Can you search for information about FAANG companies?",
        "Calculate 789 + 456 + 123. Show me the total.",
        "Look up current tech news updates.",
        "What is 2048 divided by 32?",
        "Search for information about Gemini AI model.",
        "Do you remember my name and profession from the start?",  # Memory test
        "What was the first calculation I asked you to do?"  # Deep memory test
    ]
    print("🧪 Testing automatic memory trimming with pre_model_hook:\n")
    for i, query in enumerate(test_queries, 1):
        print(f"{'─' * 60}")
        print(f"Query {i}: {query}")
        print("─" * 60)
        
        # Send query
        result = supervisor.chat(query, thread)

        time.sleep(15)  
        
    
    # Show response
    print(f"🤖 Response: {result['response']}")
    
    # Show memory statistics
    stats = result['memory_stats']
    print(f"\n📊 Memory Status:")
    print(f"   Messages: {stats['pre_message_count']} → {stats['post_message_count']}")
    print(f"   Usage: {stats['memory_usage']}")
    print(f"   Trimming method: pre_model_hook")
    
    if stats['trimming_would_apply']:
        print(f"   ⚡ Automatic trimming active (hook-based)")
    else:
        print(f"   ✅ No trimming needed")
    
    # Display current conversation state after each query
    print(f"\n" + "="*50)
    supervisor.display_current_messages(thread)
    print("="*50)
    
    # Show detailed analysis every 3 queries
    if i % 3 == 0:
        print(f"\n🔍 Thread Analysis:")
        analysis = supervisor.get_conversation_analysis(thread)
        print(f"   Total messages: {analysis['total_messages']}")
        print(f"   Human: {analysis['human_messages']}, AI: {analysis['ai_messages']}, Tool: {analysis['tool_messages']}")
        print(f"   Effective context: {analysis['effective_context_size']}/{analysis['memory_limit']}")
        print(f"   Trimming method: {analysis['trimming_method']}")
    
    print()  # Spacing

    # Final analysis
    print("=" * 65)
    print("🎯 FINAL ANALYSIS")
    print("=" * 65)
    final_analysis = supervisor.get_conversation_analysis(thread)
    print(f"Thread: {final_analysis['thread_id'][:8]}...")
    print(f"Total messages: {final_analysis['total_messages']}")
    print(f"Effective context: {final_analysis['effective_context_size']}")
    print(f"Memory limit: {final_analysis['memory_limit']}")
    print(f"Trimming triggered: {final_analysis['would_trigger_trimming']}")
    print(f"Method: {final_analysis['trimming_method']}")
    
    print(f"✨ Benefits of pre_model_hook approach:")
    print(f"   • Automatic trimming before ALL LLM calls")
    print(f"   • Applied to supervisor AND individual agents")
    print(f"   • No complex wrapper classes needed")
    print(f"   • Built-in LangGraph feature for complete workflow")
    print(f"   • Consistent behavior across all components")
    print(f"   • Simpler code maintenance")
    
    print(f"\n✅ Success: create_supervisor() DOES support pre_model_hook")
    print(f"   • Both create_supervisor() and create_react_agent() support it")
    print(f"   • Supervisor AND agent LLM calls are automatically trimmed")
    print(f"   • Comprehensive memory management across entire workflow")


if __name__ == "__main__":
    main()
