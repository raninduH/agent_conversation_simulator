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
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt.chat_agent_executor import AgentState

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

        self.summarizing_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_retries=2
        )

        self.memory = MemorySaver()
          # Create the pre-model hook for memory trimming
        self.pre_model_hook = self._create_trimming_hook()
        # Create agents and supervisor
        self.agents = self._create_agents()
        self.supervisor = self._create_supervisor()
    
    class State(AgentState):
        # NOTE: we're adding this key to keep track of previous summary information
        # to make sure we're not summarizing on every LLM call
        context: dict[str, Any]

    def _create_trimming_hook(self):
        """
        Create the pre_model_hook function for automatic memory trimming.
        """

        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.summarizing_model,
            max_tokens=600,                 # Maximum number of tokens to return in the final output (enforced after summarization)
                                            # Also used as the maximum number of tokens to feed to the summarization LLM
            
            max_tokens_before_summary=300,     # Maximum number of tokens to accumulate before triggering summarization
                                               # When cumulative message tokens reach this limit, summarization begins
                                               # Defaults to same value as max_tokens if not provided
            
            max_summary_tokens=100,            # Maximum number of tokens to budget for the summary message
                                               # Used for token budget estimation only - not enforced on the LLM
                                               # Must be less than max_tokens
            output_messages_key="llm_input_messages",
            initial_summary_prompt=ChatPromptTemplate.from_messages([
                ("placeholder", "{messages}"),
                ("user", "Create a concise summary of the conversation above, preserving key context:")
            ]),
            final_prompt=ChatPromptTemplate.from_messages([
                ("system", "Previous conversation summary: {summary}"),
                ("placeholder", "{messages}")
            ])
        )
        
        def debug_hook(state):
            """Debug wrapper with conversation flow validation."""
            messages = state.get("messages", [])
            print(f"\n🔍 DEBUG: Pre-hook messages ({len(messages)}):")
            for i, msg in enumerate(messages[-5:], 1):  # Show last 5
                print(f"   {i}. {msg.type}: {str(msg.content)[:50]}...")
            
            # Call summarization
            result = summarization_node.invoke(state)
            
            llm_input = result.get("llm_input_messages", [])
            if llm_input:
                print(f"🔍 DEBUG: Post-hook llm_input_messages ({len(llm_input)}):")
                for i, msg in enumerate(llm_input, 1):
                    print(f"   {i}. {msg.type}: {str(msg.content)[:50]}...")
                
                # ✅ Validate and fix conversation flow
                validated_messages = self._validate_conversation_flow(llm_input, messages)
                if validated_messages != llm_input:
                    print(f"🔧 DEBUG: Fixed conversation flow - adjusted {len(llm_input)} to {len(validated_messages)} messages")
                    result["llm_input_messages"] = validated_messages
            else:
                print(f"🔍 DEBUG: No llm_input_messages created (no summarization triggered)")
            
            return result
        
        return debug_hook  # ✅ Return the debug wrapper
        # def pre_model_hook(state):
        #     """Trim messages before each LLM call."""

            # messages = state.get("messages", [])

            # print("INSIDE MODEL HOOK")
            # self.print_messages_nicely(messages, title="Before Trimming")
            

            # trimmed_messages = trim_messages(
            #     messages,
            #     token_counter=len,  # <-- len will simply count the number of messages rather than tokens
            #     max_tokens=self.max_messages,  # <-- allow up to 5 messages.
            #     strategy="last",
            #     # Most chat models expect that chat history starts with either:
            #     # (1) a HumanMessage or
            #     # (2) a SystemMessage followed by a HumanMessage
            #     # start_on="human" makes sure we produce a valid chat history
            #     start_on="human",
            #     # Usually, we want to keep the SystemMessage
            #     # if it's present in the original history.
            #     # The SystemMessage has special instructions for the model.
            #     include_system=True,
            #     allow_partial=False,
            # )
            # self.print_messages_nicely(trimmed_messages, title="After Trimming")
            # print("GOING OUT OF MODEL HOOK")
            # time.sleep(20)  # Simulate processing delay for realism

            # # return {"llm_input_messages": trimmed_messages}
            # return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES)] + trimmed_messages}
        
        # return summarization_node

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
            state_schema = self.State,  # Use custom state schema for tracking
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
            pre_llm_input_messages = pre_state.values.get("llm_input_messages", [])
            self.print_messages_nicely(pre_llm_input_messages, "🧠 Pre-conversation llm_input_messages")
            pre_llm_messages_count = len(pre_llm_input_messages)
            
            pre_messages= pre_state.values.get("messages", [])
            self.print_messages_nicely(pre_messages, "🧠 Pre-conversation messages")
            pre_count = len(pre_messages)

        except:
            pre_count = 0
        
        # Create input message
        input_message = HumanMessage(content=message)
        
        print(f"💬 Processing message (pre-conversation: {pre_llm_messages_count} llm input messages)")
        print(f"💬 Processing message (pre-conversation: {pre_count} messages)")
        
        # Invoke supervisor - pre_model_hook automatically handles trimming
        result = self.supervisor.invoke(
            {"messages": [input_message]}, 
            config
        )
        
        
        # Analyze post-conversation state
        
        post_state = self.supervisor.get_state(config)
        post_llm_input_messages = post_state.values.get("llm_input_messages", [])
        self.print_messages_nicely(post_llm_input_messages, "🧠 Post-conversation llm_input_messages messages")
        
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

    def _find_proper_conversation_start(self, conversation_msgs, original_messages):
        """
        Find a proper conversation start that begins with a human message.
        
        Args:
            conversation_msgs: Current conversation messages from summarization
            original_messages: Full original message history
            
        Returns:
            Fixed conversation messages that start with human
        """
        if not conversation_msgs:
            return conversation_msgs
        
        # Get the ID of the first message in our conversation
        first_msg_id = getattr(conversation_msgs[0], 'id', None)
        if not first_msg_id:
            print("⚠️  WARNING: Message has no ID, cannot trace back to original")
            return conversation_msgs
        
        # Find this message in the original history
        first_msg_index = None
        for i, orig_msg in enumerate(original_messages):
            if getattr(orig_msg, 'id', None) == first_msg_id:
                first_msg_index = i
                break
        
        if first_msg_index is None:
            print("⚠️  WARNING: Cannot find first message in original history")
            return conversation_msgs
        
        # Look backwards to find the last human message before our conversation
        last_human_index = None
        for i in range(first_msg_index - 1, -1, -1):
            if original_messages[i].type == "human":
                last_human_index = i
                break
        
        if last_human_index is None:
            print("⚠️  WARNING: No human message found before conversation start")
            return conversation_msgs
        
        # Extract from last human message to the end of our conversation
        last_conv_msg_id = getattr(conversation_msgs[-1], 'id', None)
        last_conv_index = None
        
        for i, orig_msg in enumerate(original_messages):
            if getattr(orig_msg, 'id', None) == last_conv_msg_id:
                last_conv_index = i
                break
        
        if last_conv_index is None:
            print("⚠️  WARNING: Cannot find last conversation message in original history")
            return conversation_msgs
        
        # Extract the proper conversation segment
        proper_conversation = original_messages[last_human_index:last_conv_index + 1]
        
        print(f"🔧 DEBUG: Extended conversation from {len(conversation_msgs)} to {len(proper_conversation)} messages")
        print(f"🔧 DEBUG: Now starts with: {proper_conversation[0].type} - '{str(proper_conversation[0].content)[:50]}...'")
        
        return proper_conversation
    
    def _validate_conversation_flow(self, llm_input_messages, original_messages):
        """
        Validate and fix conversation flow to ensure Gemini compatibility.
        """
        if not llm_input_messages:
            print("⚠️  WARNING: Empty llm_input_messages, creating fallback")
            return self._create_fallback_messages(original_messages)
        
        print(f"🔍 DEBUG: Validating conversation flow...")
        
        # Separate system message (summary) from conversation messages
        system_msg = None
        conversation_msgs = []
        
        for msg in llm_input_messages:
            if msg.type == "system":
                system_msg = msg
            else:
                conversation_msgs.append(msg)
        
        # ✅ FIX: Handle case with no conversation messages
        if not conversation_msgs:
            print("⚠️  WARNING: Only system message found, adding conversation context")
            fallback_conversation = self._create_fallback_conversation(original_messages)
            
            result = []
            if system_msg:
                result.append(system_msg)
            result.extend(fallback_conversation)
            
            print(f"🔧 DEBUG: Added {len(fallback_conversation)} conversation messages to system summary")
            return result
        
        print(f"🔍 DEBUG: Found {len(conversation_msgs)} conversation messages after system message")
        
        # Check if conversation messages start with human
        first_non_system = conversation_msgs[0]
        if first_non_system.type != "human":
            print(f"⚠️  WARNING: Conversation doesn't start with human message, starts with: {first_non_system.type}")
            fixed_messages = self._find_proper_conversation_start(conversation_msgs, original_messages)
            
            result = []
            if system_msg:
                result.append(system_msg)
            result.extend(fixed_messages)
            
            print(f"🔧 DEBUG: Fixed conversation flow - now starts with: {fixed_messages[0].type if fixed_messages else 'none'}")
            return result
        
        # If everything looks good, return original
        return llm_input_messages

    def _create_fallback_conversation(self, original_messages):
        """
        Create fallback conversation when summarization leaves no conversation messages.
        """
        if not original_messages:
            return []
        
        print(f"🔧 DEBUG: Creating fallback from {len(original_messages)} original messages")
        
        # Get the last few messages including at least one human message
        recent_messages = original_messages[-8:]  # Last 8 messages
        
        # Find the last human message
        last_human_idx = None
        for i in range(len(recent_messages) - 1, -1, -1):
            if recent_messages[i].type == "human":
                last_human_idx = i
                break
        
        if last_human_idx is not None:
            # Include from last human message to end
            fallback = recent_messages[last_human_idx:]
            print(f"🔧 DEBUG: Fallback starts from human message: '{str(fallback[0].content)[:50]}...'")
            return fallback
        else:
            # If no human message found, just use the last few messages
            print(f"🔧 DEBUG: No human message found, using last 3 messages")
            return recent_messages[-3:] if len(recent_messages) >= 3 else recent_messages

    def _create_fallback_messages(self, original_messages):
        """
        Create complete fallback when llm_input_messages is empty.
        """
        print(f"🔧 DEBUG: Creating complete fallback from {len(original_messages)} messages")
        
        # Use simple trimming as complete fallback
        return trim_messages(
            original_messages,
            token_counter=len,
            max_tokens=6,  # Keep last 6 messages
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
    )

    def chat_with_stream_debug(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Chat with the supervisor using stream processing to capture llm_input_messages.
        This method captures the ephemeral llm_input_messages during execution.
        
        Args:
            message: User's input message
            thread_id: Conversation thread ID (auto-generated if None)
            
        Returns:
            Response dictionary with conversation information
        """
        import uuid
        
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
        
        # Stream the execution to capture llm_input_messages
        llm_input_captured = []
        final_result = None
        
        print(f"\n🔍 STREAMING EXECUTION:")
        for step in self.supervisor.stream({"messages": [input_message]}, config, stream_mode="debug"):
            # Check each step for llm_input_messages
            if isinstance(step, dict):
                for node_name, node_data in step.items():
                    if isinstance(node_data, dict):
                        # Check if this step contains llm_input_messages
                        if "llm_input_messages" in node_data:
                            llm_input_messages = node_data["llm_input_messages"]
                            print(f"📡 Captured llm_input_messages from {node_name}:")
                            self.print_messages_nicely(llm_input_messages, f"🔍 {node_name} llm_input_messages")
                            llm_input_captured.append({
                                "node": node_name, 
                                "messages": llm_input_messages
                            })
                        
                        # Also check for regular state updates
                        if "messages" in node_data:
                            print(f"📡 State update from {node_name}: {len(node_data['messages'])} messages")
        
        # Get final result using invoke to ensure we have the complete response
        final_result = self.supervisor.invoke({"messages": [input_message]}, config)
        
        # Analyze post-conversation state
        post_messages = final_result.get("messages", [])
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
            "hook_based_trimming": True,
            "llm_input_captured": len(llm_input_captured),
            "captured_data": llm_input_captured
        }
        
        return {
            "response": response_content,
            "thread_id": thread_id,
            "memory_stats": memory_stats
        }



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
          # Send query using stream processing to capture llm_input_messages
        result = supervisor.chat_with_stream_debug(query, thread)

        # Add delay for rate limiting
        time.sleep(25)
        
    
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
