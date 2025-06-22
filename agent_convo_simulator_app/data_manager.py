import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import random


@dataclass
class Agent:
    """Represents an agent in the conversation simulator."""
    id: str
    name: str
    role: str
    base_prompt: str
    personality_traits: List[str]
    created_at: str
    color: Optional[str] = None  # Color for the agent's messages
    
    @classmethod
    def create_new(cls, name: str, role: str, base_prompt: str, personality_traits: List[str], color: str = None) -> 'Agent':
        """Create a new agent with auto-generated ID and timestamp."""
        return cls(
            id=f"agent_{uuid.uuid4().hex[:8]}",
            name=name,
            role=role,
            base_prompt=base_prompt,
            personality_traits=personality_traits,
            created_at=datetime.now().isoformat(),
            color=color
        )


@dataclass
class Conversation:
    """Represents a conversation session."""
    id: str
    title: str
    environment: str
    scene_description: str
    agents: List[str]  # Agent IDs
    messages: List[Dict[str, Any]]
    status: str  # 'active', 'paused', 'completed'    created_at: str
    last_updated: str
    summary: Optional[str]
    thread_id: str
    agent_colors: Dict[str, str] = field(default_factory=dict)  # Maps agent names to color codes
    invocation_method: str = "round_robin"  # "round_robin" or "agent_selector"
    termination_condition: Optional[str] = None  # Condition for agent-selector to determine when to end conversation
    
    @classmethod
    def create_new(cls, title: str, environment: str, scene_description: str, agent_ids: List[str], 
                  invocation_method: str = "round_robin", termination_condition: Optional[str] = None) -> 'Conversation':
        """Create a new conversation with auto-generated ID and timestamps."""
        now = datetime.now().isoformat()
        return cls(
            id=f"conv_{uuid.uuid4().hex[:8]}",
            title=title,
            environment=environment,
            scene_description=scene_description,
            agents=agent_ids,
            messages=[],
            status='active',
            created_at=now,
            last_updated=now,
            summary=None,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            invocation_method=invocation_method,
            termination_condition=termination_condition
        )


class DataManager:
    """Manages JSON file operations for agents and conversations."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.agents_file = os.path.join(data_dir, "agents.json")
        self.conversations_file = os.path.join(data_dir, "conversations.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize JSON files with empty structures if they don't exist."""
        if not os.path.exists(self.agents_file):
            self._save_json(self.agents_file, {"agents": []})
        
        if not os.path.exists(self.conversations_file):
            self._save_json(self.conversations_file, {"conversations": []})
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, file_path: str, data: Dict[str, Any]):
        """Save JSON data to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Agent management methods
    def load_agents(self) -> List[Agent]:
        """Load all agents from JSON file."""
        data = self._load_json(self.agents_file)
        agents = []
        for agent_data in data.get("agents", []):
            agents.append(Agent(**agent_data))
        return agents
    
    def save_agent(self, agent: Agent):
        """Save a single agent to JSON file."""
        data = self._load_json(self.agents_file)
        agents = data.get("agents", [])
        
        # Update existing agent or add new one
        agent_dict = asdict(agent)
        for i, existing_agent in enumerate(agents):
            if existing_agent["id"] == agent.id:
                agents[i] = agent_dict
                break
        else:
            agents.append(agent_dict)
        
        data["agents"] = agents
        self._save_json(self.agents_file, data)
    
    def delete_agent(self, agent_id: str):
        """Delete an agent from JSON file."""
        data = self._load_json(self.agents_file)
        agents = data.get("agents", [])   
        data["agents"] = [a for a in agents if a["id"] != agent_id]
        self._save_json(self.agents_file, data)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Retrieve an agent by its ID."""
        agents = self.load_agents()
        for agent in agents:
            if agent.id == agent_id:
                return agent
        return None
    
    # Conversation management methods
    def load_conversations(self) -> List[Conversation]:
        """Load all conversations from JSON file."""
        data = self._load_json(self.conversations_file)
        conversations = []
        for conv_data in data.get("conversations", []):
            conversations.append(Conversation(**conv_data))
        return conversations
    
    def save_conversation(self, conversation: Conversation):
        """Save a single conversation to JSON file."""
        conversation.last_updated = datetime.now().isoformat()
        
        data = self._load_json(self.conversations_file)
        conversations = data.get("conversations", [])
        
        # Update existing conversation or add new one
        conv_dict = asdict(conversation)
        for i, existing_conv in enumerate(conversations):
            if existing_conv["id"] == conversation.id:
                conversations[i] = conv_dict
                break
        else:
            conversations.append(conv_dict)
        
        data["conversations"] = conversations
        self._save_json(self.conversations_file, data)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation from JSON file."""
        data = self._load_json(self.conversations_file)
        conversations = data.get("conversations", [])
        data["conversations"] = [c for c in conversations if c["id"] != conversation_id]
        self._save_json(self.conversations_file, data)
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get a specific conversation by ID."""
        conversations = self.load_conversations()
        for conversation in conversations:
            if conversation.id == conversation_id:
                return conversation
        return None
    
    def add_message_to_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """Add a message to a specific conversation."""
        conversation = self.get_conversation_by_id(conversation_id)
        if conversation:
            conversation.messages.append(message)
            self.save_conversation(conversation)
    
    def get_conversations(self) -> List[Conversation]:
        """Retrieve all conversations from the JSON file."""
        data = self._load_json(self.conversations_file)
        return [Conversation(**c) for c in data.get("conversations", [])]
    
    def update_agent_color(self, conversation_id: str, agent_name: str, color: str):
        """Update the color associated with an agent in a conversation's metadata."""
        data = self._load_json(self.conversations_file)
        conversations = data.get("conversations", [])
        
        # Find the conversation
        for conv in conversations:
            if conv.get("id") == conversation_id:
                # Initialize agent_colors if it doesn't exist
                if "agent_colors" not in conv:
                    conv["agent_colors"] = {}
                
                # Update the color
                conv["agent_colors"][agent_name] = color
                break
                
        data["conversations"] = conversations
        self._save_json(self.conversations_file, data)
    
    def get_agent_colors(self, conversation_id: str) -> Dict[str, str]:
        """Get the color mappings for agents in a conversation."""
        data = self._load_json(self.conversations_file)
        conversations = data.get("conversations", [])
        
        for conv in conversations:
            if conv.get("id") == conversation_id:
                return conv.get("agent_colors", {})
        
        return {}
