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
    api_key: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    gender: Optional[str] = "Unspecified"
    voice: Optional[str] = None
    knowledge_base: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def create_new(cls, name: str, role: str, base_prompt: str, personality_traits: List[str], 
                   api_key: str = None, tools: List[str] = None, gender: str = "Unspecified", voice: str = None) -> 'Agent':
        """Create a new agent with auto-generated ID and timestamp."""
        return cls(
            id=f"agent_{uuid.uuid4().hex[:8]}",
            name=name,
            role=role,
            base_prompt=base_prompt,
            personality_traits=personality_traits,
            created_at=datetime.now().isoformat(),
            api_key=api_key,
            tools=tools or [],
            gender=gender,
            voice=voice,
            knowledge_base=[]
        )

    @staticmethod
    def get_agent_details_by_id(agent_id: str, agents_file: str) -> dict:
        """Get agent details from agents.json by agent id."""
        with open(agents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for agent in data.get("agents", []):
            if agent.get("id") == agent_id:
                return agent
        return None


@dataclass
class Conversation:
    """Represents a conversation session."""
    id: str
    title: str
    environment: str
    scene_description: str
    agents: List[str]  # Agent IDs only
    messages: List[Dict[str, Any]]
    created_at: str
    last_updated: str
    thread_id: str
    status: str = "active"
    agent_colors: Dict[str, str] = field(default_factory=dict)
    agent_numbers: Dict[str, int] = field(default_factory=dict)
    invocation_method: str = "round_robin"
    termination_condition: Optional[str] = None
    agent_selector_api_key: Optional[str] = None
    LLM_sending_messages: List[Dict[str, Any]] = field(default_factory=list)
    voices_enabled: bool = False
    
    @classmethod
    def create_new(cls, title: str, environment: str, scene_description: str, agent_ids: List[str],
                  invocation_method: str = "round_robin", termination_condition: Optional[str] = None,
                  agent_selector_api_key: Optional[str] = None, voices_enabled: bool = False) -> 'Conversation':
        """Create a new conversation with auto-generated ID and timestamps."""
        now = datetime.now().isoformat()
        agent_numbers = {}
        for i, agent_id in enumerate(agent_ids, 1):
            agent_numbers[agent_id] = i
        return cls(
            id=f"conv_{uuid.uuid4().hex[:8]}",
            title=title,
            environment=environment,
            scene_description=scene_description,
            agents=agent_ids,
            messages=[],
            created_at=now,
            last_updated=now,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status='active',
            agent_colors={},
            agent_numbers=agent_numbers,
            invocation_method=invocation_method,
            termination_condition=termination_condition,
            agent_selector_api_key=agent_selector_api_key,
            LLM_sending_messages=[],
            voices_enabled=voices_enabled
        )

# --- New ResearchConversation dataclass ---
@dataclass
class ResearchConversation:
    """Represents a research group conversation session."""
    id: str
    research_name: str
    research_problem: str
    extra_consider: str
    research_goal: str
    agents: List[str]  # Agent IDs only
    messages: List[Dict[str, Any]]
    created_at: str
    last_updated: str
    thread_id: str
    status: str = "active"
    voices_enabled: bool = False
    agent_colors: Dict[str, str] = field(default_factory=dict)
    agent_numbers: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def create_new(cls, research_name: str, research_problem: str, extra_consider: str, research_goal: str, agent_ids: List[str], voices_enabled: bool = False, agent_colors: Dict[str, str] = None, agent_numbers: Dict[str, int] = None) -> 'ResearchConversation':
        """Create a new research conversation with auto-generated ID and timestamps."""
        now = datetime.now().isoformat()
        return cls(
            id=f"research_{uuid.uuid4().hex[:8]}",
            research_name=research_name,
            research_problem=research_problem,
            extra_consider=extra_consider,
            research_goal=research_goal,
            agents=agent_ids,
            messages=[],
            created_at=now,
            last_updated=now,
            thread_id=f"thread_{uuid.uuid4().hex[:8]}",
            status='active',
            voices_enabled=voices_enabled,
            agent_colors=agent_colors or {},
            agent_numbers=agent_numbers or {}
        )
    
class DataManager:


    """Manages JSON file operations for agents and conversations."""
    
    def __init__(self):
        self.data_dir = os.path.dirname(__file__)
        self.agents_file = os.path.join(self.data_dir, "agents.json")
        self.conversations_file = os.path.join(self.data_dir, "conversations.json")
        # Add research_conversations_file in memory/ subfolder
        self.research_conversations_file = os.path.join(self.data_dir, "memory", "research_conversations.json")

        # Caching
        self._agents_cache = None
        self._agents_cache_timestamp = None

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(self.data_dir), "memory"), exist_ok=True)

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
    def load_agents(self, force_reload: bool = False) -> List[Agent]:
        """Load all agents from JSON file with caching."""
        # Check if we need to reload
        try:
            file_mtime = os.path.getmtime(self.agents_file)
        except OSError:
            file_mtime = 0
        
        # Use cache if available and file hasn't changed, unless force_reload is True
        if (not force_reload and 
            self._agents_cache is not None and 
            self._agents_cache_timestamp is not None and 
            file_mtime <= self._agents_cache_timestamp):
            return self._agents_cache
        
        # Load from file
        data = self._load_json(self.agents_file)
        agents = []
        for agent_data in data.get("agents", []):
            # Remove any extra keys not in Agent dataclass
            allowed_keys = {f.name for f in Agent.__dataclass_fields__.values()}
            filtered_agent_data = {k: v for k, v in agent_data.items() if k in allowed_keys}
            # Ensure knowledge_base field exists, default to empty list if not present
            if 'knowledge_base' not in filtered_agent_data:
                filtered_agent_data['knowledge_base'] = []
            agent = Agent(**filtered_agent_data)
            agents.append(agent)
        
        # Update cache
        self._agents_cache = agents
        self._agents_cache_timestamp = file_mtime
        
        return agents
    
    def save_agent(self, agent: Agent):
        """Save a single agent to JSON file."""
        data = self._load_json(self.agents_file)
        agents = data.get("agents", [])
        
        # Update existing agent or add new one
        agent_dict = asdict(agent)
        # Remove 'color' key if present
        if 'color' in agent_dict:
            del agent_dict['color']
        for i, existing_agent in enumerate(agents):
            # Remove 'color' key from existing agent dicts
            if 'color' in existing_agent:
                del existing_agent['color']
            if existing_agent["id"] == agent.id:
                agents[i] = agent_dict
                break
        else:
            agents.append(agent_dict)
        
        data["agents"] = agents
        self._save_json(self.agents_file, data)
        
        # Invalidate cache
        self._agents_cache = None
        self._agents_cache_timestamp = None
    
    def delete_agent(self, agent_id: str):
        """Delete an agent from JSON file."""
        data = self._load_json(self.agents_file)
        agents = data.get("agents", [])   
        data["agents"] = [a for a in agents if a["id"] != agent_id]
        self._save_json(self.agents_file, data)
        
        # Invalidate cache
        self._agents_cache = None
        self._agents_cache_timestamp = None
    
    def remove_document_from_knowledge_base(self, agent_id: str, doc_name: str) -> bool:
        """
        Remove a document (by doc_name) from the knowledge_base list of the agent with the given agent_id.
        Returns True if removed, False if not found or agent not found.
        """
        agent = self.get_agent_by_id(agent_id)
        if not agent or not hasattr(agent, 'knowledge_base'):
            return False
        original_len = len(agent.knowledge_base)
        agent.knowledge_base = [doc for doc in agent.knowledge_base if doc.get('doc_name') != doc_name]
        if len(agent.knowledge_base) < original_len:
            self.save_agent(agent)
            return True
        return False

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Retrieve an agent by its ID."""
        agents = self.load_agents()
        for agent in agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_all_agent_ids(self) -> list:
        """Return a list of all agent IDs from agents.json."""
        agents = self.load_agents()
        return [agent.id for agent in agents]
    
    # Conversation management methods
    def load_conversations(self) -> List[Conversation]:
        """Load all conversations from JSON file."""
        data = self._load_json(self.conversations_file)
        conversations = []
        for conv_data in data.get("conversations", []):
            try:
                # Remove any keys not in Conversation dataclass
                allowed_keys = {f.name for f in Conversation.__dataclass_fields__.values()}
                filtered_conv_data = {k: v for k, v in conv_data.items() if k in allowed_keys}
                # Fix legacy field name
                if 'agent_sending_messages' in filtered_conv_data:
                    filtered_conv_data['LLM_sending_messages'] = filtered_conv_data.pop('agent_sending_messages')
                # Ensure required fields
                for key in allowed_keys:
                    if key not in filtered_conv_data:
                        if Conversation.__dataclass_fields__[key].default_factory is not None:
                            filtered_conv_data[key] = Conversation.__dataclass_fields__[key].default_factory()
                        else:
                            filtered_conv_data[key] = Conversation.__dataclass_fields__[key].default if Conversation.__dataclass_fields__[key].default is not None else None
                conversations.append(Conversation(**filtered_conv_data))
            except Exception as e:
                print(f"Error loading conversation {conv_data.get('id', 'unknown')}: {e}")
        return conversations
    
    def save_conversation(self, conversation: Conversation):
        """Save a single conversation to JSON file."""
        conversation.last_updated = datetime.now().isoformat()
        
        data = self._load_json(self.conversations_file)
        conversations = data.get("conversations", [])
        
        # Update existing conversation or add new one
        conv_dict = asdict(conversation)
        # Remove agent_temp_numbers if present
        if "agent_temp_numbers" in conv_dict:
            del conv_dict["agent_temp_numbers"]
        data_key = "agent_numbers"
        # Ensure agent_numbers is present
        if "agent_numbers" not in conv_dict and hasattr(conversation, "agent_numbers"):
            conv_dict["agent_numbers"] = getattr(conversation, "agent_numbers", {})
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
    
    def add_message_to_conversation(self, conversation_id, message):
        """Add a message to a conversation and save to disk."""
        # Remove timestamp and message_id if present
        message.pop('timestamp', None)
        message.pop('message_id', None)
        
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
    
    def test_method_exists(self):
        """Test method to verify the class is working properly."""
        print(f"DataManager methods: {[method for method in dir(self) if not method.startswith('_')]}")
        return True
    
    def clear_agents_cache(self):
        """Manually clear the agents cache to force reload on next access."""
        self._agents_cache = None
        self._agents_cache_timestamp = None
        
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
    
    def reassign_voices_for_conversation(self, conversation_id: str, agent_ids: List[str]):
        """Reassign voices when agents in a conversation are modified."""
        conversation = self.get_conversation_by_id(conversation_id)
        if not conversation or not conversation.voices_enabled:
            return
        
        # Get current agent objects
        agents = [self.get_agent_by_id(agent_id) for agent_id in agent_ids]
        agents = [agent for agent in agents if agent]  # Filter out None values
        
        if not agents:
            return
        
        # Import voice manager
        from voice_assignment import VoiceAssignmentManager
        voice_manager = VoiceAssignmentManager()
        
        # Keep existing voice assignments for agents that are still in the conversation
        existing_assignments = {}
        for agent_id in agent_ids:
            if agent_id in conversation.agent_voices:
                existing_assignments[agent_id] = conversation.agent_voices[agent_id]
        
        # Reassign voices for all agents
        new_assignments = voice_manager.assign_voices_to_agents(agents, existing_assignments)
        conversation.agent_voices = new_assignments
        
        # Save the updated conversation
        self.save_conversation(conversation)
        
        print(f"DEBUG: Reassigned voices for conversation {conversation_id}: {new_assignments}")


    # --- Research Conversation management methods ---
    def load_research_conversations(self) -> List[ResearchConversation]:
        """Load all research conversations from JSON file."""
        data = self._load_json(self.research_conversations_file)
        research_conversations = []
        for conv_data in data.get("research_conversations", []):
            try:
                allowed_keys = {f.name for f in ResearchConversation.__dataclass_fields__.values()}
                filtered_conv_data = {k: v for k, v in conv_data.items() if k in allowed_keys}
                for key in allowed_keys:
                    if key not in filtered_conv_data:
                        if ResearchConversation.__dataclass_fields__[key].default_factory is not None:
                            filtered_conv_data[key] = ResearchConversation.__dataclass_fields__[key].default_factory()
                        else:
                            filtered_conv_data[key] = ResearchConversation.__dataclass_fields__[key].default if ResearchConversation.__dataclass_fields__[key].default is not None else None
                research_conversations.append(ResearchConversation(**filtered_conv_data))
            except Exception as e:
                print(f"Error loading research conversation {conv_data.get('id', 'unknown')}: {e}")
        return research_conversations

    def save_research_conversation(self, research_conversation: ResearchConversation):
        """Save a single research conversation to JSON file."""
        research_conversation.last_updated = datetime.now().isoformat()
        data = self._load_json(self.research_conversations_file)
        conversations = data.get("research_conversations", [])
        conv_dict = asdict(research_conversation)
        for i, existing_conv in enumerate(conversations):
            if existing_conv["id"] == research_conversation.id:
                conversations[i] = conv_dict
                break
        else:
            conversations.append(conv_dict)
        data["research_conversations"] = conversations
        self._save_json(self.research_conversations_file, data)

    def delete_research_conversation(self, research_id: str):
        """Delete a research conversation from JSON file."""
        data = self._load_json(self.research_conversations_file)
        conversations = data.get("research_conversations", [])
        data["research_conversations"] = [c for c in conversations if c["id"] != research_id]
        self._save_json(self.research_conversations_file, data)

    def get_research_conversation_by_id(self, research_id: str) -> Optional[ResearchConversation]:
        """Get a specific research conversation by ID."""
        conversations = self.load_research_conversations()
        for conversation in conversations:
            if conversation.id == research_id:
                return conversation
        return None

    def add_message_to_research_conversation(self, research_id, message):
        """Add a message to a research conversation and save to disk."""
        message.pop('timestamp', None)
        message.pop('message_id', None)
        conversation = self.get_research_conversation_by_id(research_id)
        if conversation:
            conversation.messages.append(message)
            self.save_research_conversation(conversation)

    def get_research_conversations(self) -> List[ResearchConversation]:
        """Retrieve all research conversations from the JSON file."""
        data = self._load_json(self.research_conversations_file)
        return [ResearchConversation(**c) for c in data.get("research_conversations", [])]