"""
Simplified Mock Conversation Engine for Testing
This provides a basic implementation that works without LangGraph dependencies
for initial testing and development purposes.
"""

import os
import time
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading


class MockConversationSimulatorEngine:
    """
    Mock conversation simulator for testing without LangGraph dependencies.
    This simulates agent conversations using simple templates and random responses.
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize the mock conversation simulator engine."""
        self.google_api_key = google_api_key
        self.active_conversations = {}
        self.message_callbacks = {}
        
        # Mock response templates for different agent personalities
        self.response_templates = {
            "curious": [
                "That's really interesting! Can you tell me more about {topic}?",
                "I'm curious about {topic}. What do you think about it?",
                "Fascinating! How did you come to that conclusion about {topic}?",
                "I wonder if there's another way to look at {topic}?",
            ],
            "enthusiastic": [
                "Oh wow, {topic} is amazing! I love discussing this!",
                "That's fantastic! I'm so excited to explore {topic} with you all!",
                "This is great! {topic} always gets me thinking!",
                "I absolutely love how you explained {topic}!",
            ],
            "analytical": [
                "Let me think about {topic} from a logical perspective...",
                "The data suggests that {topic} has several interesting aspects.",
                "If we analyze {topic} systematically, we can see...",
                "From an analytical standpoint, {topic} presents these considerations...",
            ],
            "creative": [
                "You know, {topic} reminds me of a story I once heard...",
                "What if we imagined {topic} in a completely different way?",
                "I see {topic} like a painting with many colors and textures.",
                "There's something poetic about how you described {topic}.",
            ],
            "diplomatic": [
                "I appreciate both perspectives on {topic}.",
                "That's a valid point about {topic}. What do others think?",
                "I can see the merit in different approaches to {topic}.",
                "Perhaps we can find common ground on {topic}?",
            ]
        }
        
        # Mock conversation starters
        self.conversation_starters = [
            "What brings everyone here today?",
            "I've been thinking about our current environment. What do you notice?",
            "This is such an interesting place to have a conversation.",
            "I'm excited to hear everyone's thoughts and perspectives.",
            "What's been on your mind lately?",
        ]
    
    def start_conversation(self, conversation_id: str, agents_config: List[Dict[str, str]], 
                          environment: str, scene_description: str, initial_message: str = None) -> str:
        """Start a new mock conversation session."""
        thread_id = f"thread_{conversation_id}"
        
        # Store conversation data
        self.active_conversations[conversation_id] = {
            "thread_id": thread_id,
            "environment": environment,
            "scene_description": scene_description,
            "agents_config": agents_config,
            "status": "active",
            "message_count": 0,
            "last_speaker": None
        }
        
        # Start conversation with initial message or auto-start
        if initial_message:
            # Schedule the first response after a short delay
            threading.Timer(2.0, self._generate_response, args=(conversation_id, initial_message)).start()
        else:
            # Start with a conversation starter
            starter = random.choice(self.conversation_starters)
            threading.Timer(1.0, self._send_agent_message, 
                          args=(conversation_id, starter, random.choice(agents_config)["name"])).start()
        
        return thread_id
    
    def send_message(self, conversation_id: str, message: str, sender: str = "user") -> Dict[str, Any]:
        """Send a message to the mock conversation and generate response."""
        if conversation_id not in self.active_conversations:
            return {"success": False, "error": "Conversation not found"}
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return {"success": False, "error": "Conversation is not active"}
        
        try:
            # Update message count
            conv_data["message_count"] += 1
            
            # Schedule a response after a random delay (1-4 seconds)
            delay = random.uniform(1.0, 4.0)
            threading.Timer(delay, self._generate_response, args=(conversation_id, message)).start()
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_response(self, conversation_id: str, trigger_message: str):
        """Generate a mock agent response."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        # Select an agent to respond (avoid same agent responding twice in a row)
        available_agents = conv_data["agents_config"].copy()
        if conv_data["last_speaker"]:
            available_agents = [a for a in available_agents if a["name"] != conv_data["last_speaker"]]
        
        if not available_agents:
            available_agents = conv_data["agents_config"]
        
        responding_agent = random.choice(available_agents)
        conv_data["last_speaker"] = responding_agent["name"]
        
        # Generate response based on agent personality and trigger
        response = self._generate_agent_response(responding_agent, trigger_message, conv_data)
        
        # Send the response
        self._send_agent_message(conversation_id, response, responding_agent["name"])
        
        # Maybe generate another response from a different agent (30% chance)
        if random.random() < 0.3 and len(conv_data["agents_config"]) > 1:
            delay = random.uniform(2.0, 5.0)
            threading.Timer(delay, self._generate_followup_response, 
                          args=(conversation_id, response)).start()
    
    def _generate_followup_response(self, conversation_id: str, previous_message: str):
        """Generate a follow-up response from a different agent."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        # Select a different agent
        available_agents = [a for a in conv_data["agents_config"] 
                          if a["name"] != conv_data["last_speaker"]]
        
        if not available_agents:
            return
        
        responding_agent = random.choice(available_agents)
        conv_data["last_speaker"] = responding_agent["name"]
        
        # Generate a follow-up response
        response = self._generate_agent_response(responding_agent, previous_message, conv_data, is_followup=True)
        
        # Send the response
        self._send_agent_message(conversation_id, response, responding_agent["name"])
    
    def _generate_agent_response(self, agent_config: Dict[str, str], trigger_message: str, 
                                conv_data: Dict[str, Any], is_followup: bool = False) -> str:
        """Generate a response based on agent personality."""
        # Determine agent personality from traits or role
        personality = self._determine_personality(agent_config)
        
        # Extract topic from trigger message (simple keyword extraction)
        topic = self._extract_topic(trigger_message, conv_data)
        
        # Select appropriate template
        if personality in self.response_templates:
            template = random.choice(self.response_templates[personality])
        else:
            template = random.choice(self.response_templates["curious"])
        
        # Generate response
        if is_followup:
            followup_starters = [
                "I agree with that perspective on {topic}.",
                "That makes me think about {topic} differently.",
                "Building on what was just said about {topic}...",
                "I have a different take on {topic}.",
            ]
            template = random.choice(followup_starters)
        
        # Fill in the template
        response = template.format(topic=topic)
        
        # Add personality-specific additions
        if personality == "enthusiastic" and random.random() < 0.3:
            response += " This is exactly the kind of discussion I love!"
        elif personality == "analytical" and random.random() < 0.3:
            response += " We should consider all the variables involved."
        elif personality == "creative" and random.random() < 0.3:
            response += " There are so many creative possibilities here!"
        
        return response
    
    def _determine_personality(self, agent_config: Dict[str, str]) -> str:
        """Determine agent personality from configuration."""
        # Check base prompt for personality keywords
        prompt = agent_config.get("base_prompt", "").lower()
        role = agent_config.get("role", "").lower()
        
        if any(word in prompt for word in ["curious", "question", "explore"]):
            return "curious"
        elif any(word in prompt for word in ["enthusiastic", "excited", "energetic"]):
            return "enthusiastic"
        elif any(word in prompt for word in ["analytical", "logical", "systematic"]):
            return "analytical"
        elif any(word in prompt for word in ["creative", "imaginative", "artistic"]):
            return "creative"
        elif any(word in prompt for word in ["diplomatic", "balanced", "thoughtful"]):
            return "diplomatic"
        elif "teacher" in role or "professor" in role:
            return "analytical"
        elif "artist" in role or "writer" in role:
            return "creative"
        else:
            return "curious"  # Default
    
    def _extract_topic(self, message: str, conv_data: Dict[str, Any]) -> str:
        """Extract a topic from the message for response generation."""
        # Simple topic extraction - look for key nouns or use environment
        words = message.lower().split()
        
        # Common topics that might come up
        topics = ["life", "work", "art", "science", "books", "music", "travel", "food", 
                 "technology", "nature", "philosophy", "relationships", "learning"]
        
        # Look for topics in the message
        for topic in topics:
            if topic in words:
                return topic
        
        # If no specific topic found, use environment or generic topics
        environment = conv_data.get("environment", "").lower()
        if "coffee" in environment or "café" in environment:
            return random.choice(["coffee", "conversation", "this place"])
        elif "library" in environment:
            return random.choice(["books", "learning", "knowledge"])
        elif "park" in environment:
            return random.choice(["nature", "outdoors", "this beautiful day"])
        else:
            return random.choice(["this topic", "our conversation", "what you mentioned"])
    
    def _send_agent_message(self, conversation_id: str, message: str, agent_name: str):
        """Send an agent message through the callback system."""
        if conversation_id in self.message_callbacks:
            message_data = {
                "sender": agent_name,
                "content": message,
                "type": "ai",
                "timestamp": datetime.now().isoformat()
            }
            self.message_callbacks[conversation_id](message_data)
    
    def pause_conversation(self, conversation_id: str):
        """Pause a conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "paused"
    
    def resume_conversation(self, conversation_id: str):
        """Resume a paused conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "active"
            # Maybe generate a "resuming" message
            conv_data = self.active_conversations[conversation_id]
            if conv_data["agents_config"]:
                agent = random.choice(conv_data["agents_config"])
                resume_messages = [
                    "Alright, where were we?",
                    "Let's continue our discussion.",
                    "I'm ready to keep talking!",
                    "This conversation is really interesting."
                ]
                message = random.choice(resume_messages)
                threading.Timer(1.0, self._send_agent_message, 
                              args=(conversation_id, message, agent["name"])).start()
    
    def stop_conversation(self, conversation_id: str):
        """Stop and remove a conversation from active sessions."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        if conversation_id in self.message_callbacks:
            del self.message_callbacks[conversation_id]
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate a mock conversation summary."""
        if conversation_id not in self.active_conversations:
            return "Conversation not found"
        
        conv_data = self.active_conversations[conversation_id]
        agents = [agent["name"] for agent in conv_data["agents_config"]]
        
        summary = f"""Conversation Summary:
        
Environment: {conv_data['environment']}
Participants: {', '.join(agents)}
Messages exchanged: {conv_data['message_count']}
Status: {conv_data['status']}

The conversation has been taking place in {conv_data['environment']}. 
The participants ({', '.join(agents)}) have been engaging in discussion 
about various topics related to their environment and interests. 
{conv_data['message_count']} messages have been exchanged so far.

This is a mock summary generated for testing purposes."""
        
        return summary
    
    def register_message_callback(self, conversation_id: str, callback: Callable):
        """Register a callback function for real-time message updates."""
        self.message_callbacks[conversation_id] = callback
    
    def change_scene(self, conversation_id: str, new_environment: str, new_scene_description: str):
        """Change the scene/environment for an active conversation."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conv_data = self.active_conversations[conversation_id]
        
        # Update environment data
        conv_data["environment"] = new_environment
        conv_data["scene_description"] = new_scene_description
        
        # Generate a response to the scene change
        if conv_data["agents_config"] and conv_data["status"] == "active":
            agent = random.choice(conv_data["agents_config"])
            scene_responses = [
                f"Wow, what a change! I really like this new {new_environment}.",
                f"This {new_environment} has such a different atmosphere!",
                f"I'm excited to continue our conversation in this new setting.",
                f"The {new_environment} really changes the mood, doesn't it?",
            ]
            response = random.choice(scene_responses)
            threading.Timer(2.0, self._send_agent_message, 
                          args=(conversation_id, response, agent["name"])).start()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
