"""
Configuration settings for the Agent Conversation Simulator.
"""

# UI color settings
UI_COLORS = {
    # Background color for the application
    "app_background": "#E5D9F2",
    
    # Chat bubble colors for user and system messages
    "user_bubble": "#DCF8C6",
    "system_bubble": "#F0F0F0",
    "ai_bubble": "#CAE8BD",  # Default AI bubble color
    
    # List of colors for agent chat bubbles (each agent gets a unique color)
    "agent_colors": [
        "#CAE8BD",  # Light green
        "#FFE8CD",  # Light orange
        "#4F8B61",  # Light purple
        "#FFDCDC",  # Light pink
        "#A2AADB",  # Light blue
        "#FADA7A",  # Light yellow
        "#D4F6FF",  # Light cyan
        "#FCFAEE"   # Light cream
    ],
      # Background color for the chat area
    "chat_background": "#E5D9F2"
}

# Conversation timing settings (in seconds)
CONVERSATION_TIMING = {
    # Delay before starting the conversation cycle
    "start_delay": 1.0,
    
    # Delay between agent turns (min and max for random selection)
    "agent_turn_delay_min": 5.0,
    "agent_turn_delay_max": 10.0,
    
    # Delay when resuming a paused conversation
    "resume_delay": 1.0,
    
    # Delay when an error occurs and trying to continue
    "error_retry_delay": 30.0
}

# Message handling settings
MESSAGE_SETTINGS = {
    # Maximum number of messages before summarization
    "max_messages_before_summary": 20,
    
    # Number of recent messages to keep after summarization
    "messages_to_keep_after_summary": 10
}

# Agent settings
AGENT_SETTINGS = {
    # Temperature for agent responses (0.0 to 1.0)
    "response_temperature": 0.7,
    
    # Temperature for summarization (0.0 to 1.0, lower is more consistent)
    "summary_temperature": 0.3,
    
    # Maximum retries for API calls
    "max_retries": 2,
    
    # Frequency of reminding agents about termination condition
    # (every X invocations for each agent)
    "termination_reminder_frequency": 4,
    
    # Timeout for parallel agent responses in human-like-chat mode (seconds)
    "parallel_response_timeout": 30.0
}

# Gemini model settings
MODEL_SETTINGS = {
    # Model name for agent responses
    "agent_model": "gemini-2.0-flash-exp",
    
    # Model name for summarization (can be different for cost optimization)
    "summary_model": "gemini-2.0-flash-exp"
}
