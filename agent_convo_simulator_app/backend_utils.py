
import os
from typing import Dict, List
from langchain_core.messages import HumanMessage
from .config import MESSAGE_SETTINGS, AGENT_SETTINGS, MODEL_SETTINGS
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# Only use the environment variable for the summary model
summary_api_key = os.getenv("GOOGLE_API_KEY")
if not summary_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required for the summary model.")


summary_model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["summary_model"],
            temperature=AGENT_SETTINGS["summary_temperature"],            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=summary_api_key
        )

def message_list_summarization(messages: List[Dict[str, str]], 
                                no_of_messages_to_trigger_summarization: int = None) -> List[Dict[str, str]]:
    """
    Summarize messages when they exceed the trigger threshold.
    
    Args:
        messages: List of message dictionaries
        no_of_messages_to_trigger_summarization: Threshold for triggering summarization (defaults to config value)
        
    Returns:
        Updated messages list with summary and last N messages (N from config)
    """
    # Use config values if not specified
    if no_of_messages_to_trigger_summarization is None:
        no_of_messages_to_trigger_summarization = MESSAGE_SETTINGS["max_messages_before_summary"]
    
    messages_to_keep = MESSAGE_SETTINGS["messages_to_keep_after_summary"]
    
    if len(messages) <= no_of_messages_to_trigger_summarization:
        return messages
    
    # Check if there's already a summary at the beginning
    has_existing_summary = (messages and 
                            len(messages) > 0 and 
                            "past_convo_summary" in messages[0])
    
    if has_existing_summary:
        # Get existing summary and messages to summarize
        existing_summary = messages[0]["past_convo_summary"]
        messages_to_summarize = messages[1:-messages_to_keep]  # Exclude summary and last N
        last_n_messages = messages[-messages_to_keep:]
    else:
        # No existing summary
        existing_summary = None
        messages_to_summarize = messages[:-messages_to_keep]  # All except last N
        last_n_messages = messages[-messages_to_keep:]
    
    # Create summarization prompt
    if existing_summary:
        summary_prompt = f"Previous conversation summary: {existing_summary}\n\nRecent conversation messages:\n"
    else:
        summary_prompt = "Conversation messages to summarize:\n"
        # Add messages to summarize
    for msg in messages_to_summarize:
        if "agent_name" in msg and "message" in msg:
            summary_prompt += f"{msg['agent_name']}: {msg['message']}\n"
    
    summary_prompt += "\nPlease provide a concise summary of the conversation above, capturing the key topics, main points discussed, and important context. Only return the summary text, nothing else."
    
    try:
        # Get summary from LLM
        response = summary_model.invoke([HumanMessage(content=summary_prompt)])
        new_summary = response.content.strip()
        
        # Create new messages list with summary + last N messages            
        new_messages = [{"past_convo_summary": new_summary}] + last_n_messages
        
        return new_messages
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Fallback: just keep last N+5 messages if summarization fails
        fallback_count = messages_to_keep + 5
        return messages[-fallback_count:]
    

def create_agent_base_prompt(agent_config):
    """
    Create a simple base prompt for an agent using their name, personality traits, role, and base prompt.
    """
    agent_name = agent_config["name"]
    personality_traits = agent_config["personality_traits"]
    role = agent_config["role"]
    base_prompt = agent_config["base_prompt"]
    prompt = (
        f"you are {agent_name}, your personality traits are {personality_traits} and your role in this conversation is {role}. "
        f"The following is your character description prompt {base_prompt}.\n\n"
        "Always answer based on the above characteristics. Stay in character always. "
    )
    return prompt


def create_agent_prompt(agent_config, environment, scene_description, messages, all_agents, termination_condition=None, should_remind_termination=False, conversation_id=None, agent_name=None, available_tools=None, agent_obj=None):
    """
    Create the prompt for an agent including scene, participants, and conversation history.
    """
    if not agent_name:
        agent_name = agent_config["name"]
    agent_role = agent_config["role"]
    base_prompt = agent_config["base_prompt"]
    prompt = f"""You are {agent_name}, a {agent_role}.
            \n{base_prompt}
            Always answer based on the above characteristics. Stay in character always.
            INITIAL SCENE: {environment}
            SCENE DESCRIPTION: {scene_description}
            \nPARTICIPANTS: {', '.join(all_agents)}\n\nTool Usage: Use your tools freely in the first instance you feel,  just like a noraml person using their mobile phone as a tool. No need to get permsission from other agents. But when it's necessary discuss with other agents how the tools should be used.\n\n"""
    
    # Always use the current messages list as the single source of truth
    if messages:
        if messages[0].get("past_convo_summary"):
            prompt += f"PREVIOUS CONVERSATION SUMMARY: {messages[0]['past_convo_summary']}\n\n"
            recent_messages = messages[1:]
        else:
            recent_messages = messages
        if recent_messages:
            prompt += "CONVERSATION SO FAR:\n"
            for msg in recent_messages:
                if "agent_name" in msg and "message" in msg:
                    prompt += f"{msg['agent_name']}: {msg['message']}\n"
            prompt += "\n"
    if should_remind_termination and termination_condition:
        prompt += f"""TERMINATION CONDITION REMINDER: The conversation should end when the following condition is met:\n{termination_condition}\n\nKeep this condition in mind while participating in the conversation. Naturally deviate the conversation into the direction where the condition will be met. and stay true to your personality traits.\n\n"""
    if available_tools:
        prompt += f"""AVAILABLE TOOLS: You have access to the following tools: {', '.join(available_tools)}\nUse these tools when they can help you respond more effectively to the conversation.\nOnly use tools when they are relevant to the current conversation context.\nDon't mention the tools explicitly unless asked about your capabilities.\n\n"""
    if agent_obj and hasattr(agent_obj, 'knowledge_base') and agent_obj.knowledge_base:
        knowledge_descriptions = []
        for doc in agent_obj.knowledge_base:
            if hasattr(doc, 'metadata') and 'description' in doc.metadata:
                knowledge_descriptions.append(doc.metadata['description'])
        prompt += f"""PERSONAL KNOWLEDGE BASE: You have access to a personal knowledge base containing the following documents:\n{chr(10).join(knowledge_descriptions)}\n\nUse the knowledge_base_retriever tool to search through these documents when relevant to the conversation. \nThis knowledge base contains specialized information that can help you stay true to your role and provide more informed responses.\nOnly search your knowledge base when the conversation topic relates to the content of your documents.\n\n"""
    prompt += f"""Give your response to the ongoing conversation as {agent_name}. \nKeep your response natural, conversational, and true to your character. Always respons with the charateristics/personality of your character. \nRespond as if you're speaking directly in the conversation (don't say \"As {agent_name}, I would say...\" just respond naturally).\nRespond only to the dialog parts said by the other agents.\nKeep responses to 1-3 sentences to maintain good conversation flow."""
    return prompt



def _load_agent_tools(agent_name):
    # Loads tools for a specific agent based on their configuration.
    from .tools import (
        get_browser_tools
    )

    import importlib


    browser_manipulation_toolkit = []
    try:
        browser_manipulation_toolkit =  get_browser_tools()
            
    except Exception:
        pass
    
    # Load agent tool names from agents.json
    import json, os
    agents_json_path = os.path.join(os.path.dirname(__file__), "agents.json")
    with open(agents_json_path, "r", encoding="utf-8") as f:
        agents_data = json.load(f)
    agent_tool_names = []
    for agent in agents_data.get("agents", []):
        if agent.get("name") == agent_name:
            agent_tool_names = agent.get("tools", [])
            break
    loaded_tools = []

    for tool_name in agent_tool_names:
        if tool_name == "browser_manipulation_toolkit":
            loaded_tools.extend(browser_manipulation_toolkit)
            continue
        try:
            tools_module = importlib.import_module(".tools", __package__)
            tool_func = getattr(tools_module, tool_name)
            loaded_tools.append(tool_func)
            
        except (ImportError, AttributeError) as e:
            print(f"Warning: Tool '{tool_name}' for agent '{agent_name}' not found: {e}")

        
    return loaded_tools


