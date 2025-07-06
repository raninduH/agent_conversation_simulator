"""
Voice Assignment Manager
Handles assignment of voices to agents based on gender and availability
"""

import json
import os
import random
from typing import Dict, List, Optional
from data_manager import Agent


class VoiceAssignmentManager:
    """Manages voice assignment for agents based on gender."""
    
    def __init__(self, voices_file_path: str = None):
        """Initialize with path to voices JSON file."""
        if voices_file_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            voices_file_path = os.path.join(current_dir, "kokoro_voices.json")
        
        self.voices_file_path = voices_file_path
        self.voices = self._load_voices()
    
    def _load_voices(self) -> Dict[str, List[str]]:
        """Load available voices from JSON file."""
        try:
            with open(self.voices_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load voices from {self.voices_file_path}: {e}")
            # Return default voices if file not found
            return {
                "female": ["af_alloy", "af_aoede", "af_bella", "af_heart", 
                          "af_jessica", "af_kore", "af_nicole", "af_nova", 
                          "af_river", "af_sarah", "af_sky"],
                "male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                        "am_michael", "am_onyx", "am_puck", "am_santa"]
            }
    
    def assign_voices_to_agents(self, agents: List[Agent], existing_assignments: Dict[str, str] = None) -> Dict[str, str]:
        """
        Assign voices to agents based on their gender, avoiding duplicates when possible.
        
        Args:
            agents: List of Agent objects to assign voices to
            existing_assignments: Dict of agent_id -> voice_name for already assigned voices
            
        Returns:
            Dict mapping agent_id to assigned voice name
        """
        if existing_assignments is None:
            existing_assignments = {}
        
        voice_assignments = existing_assignments.copy()
        
        # Separate agents by gender
        male_agents = [agent for agent in agents if agent.gender.lower() == "male"]
        female_agents = [agent for agent in agents if agent.gender.lower() == "female"]
        unspecified_agents = [agent for agent in agents if agent.gender.lower() not in ["male", "female"]]
        
        # Get available voices
        available_male_voices = self.voices.get("male", []).copy()
        available_female_voices = self.voices.get("female", []).copy()
        
        # Remove already assigned voices to avoid duplicates
        for agent_id, voice in existing_assignments.items():
            if voice in available_male_voices:
                available_male_voices.remove(voice)
            elif voice in available_female_voices:
                available_female_voices.remove(voice)
        
        # Assign voices to male agents
        for agent in male_agents:
            if agent.id not in voice_assignments:
                if available_male_voices:
                    voice = random.choice(available_male_voices)
                    voice_assignments[agent.id] = voice
                    available_male_voices.remove(voice)
                else:
                    # If we run out of unique voices, use a random one
                    voice_assignments[agent.id] = random.choice(self.voices.get("male", ["am_adam"]))
        
        # Assign voices to female agents
        for agent in female_agents:
            if agent.id not in voice_assignments:
                if available_female_voices:
                    voice = random.choice(available_female_voices)
                    voice_assignments[agent.id] = voice
                    available_female_voices.remove(voice)
                else:
                    # If we run out of unique voices, use a random one
                    voice_assignments[agent.id] = random.choice(self.voices.get("female", ["af_alloy"]))
        
        # Assign voices to unspecified gender agents (use available voices from either gender)
        all_available_voices = available_male_voices + available_female_voices
        for agent in unspecified_agents:
            if agent.id not in voice_assignments:
                if all_available_voices:
                    voice = random.choice(all_available_voices)
                    voice_assignments[agent.id] = voice
                    all_available_voices.remove(voice)
                else:
                    # If we run out of unique voices, use any random voice
                    all_voices = self.voices.get("male", []) + self.voices.get("female", [])
                    voice_assignments[agent.id] = random.choice(all_voices) if all_voices else "am_adam"
        
        return voice_assignments
    
    def get_voice_for_agent(self, agent_id: str, voice_assignments: Dict[str, str]) -> Optional[str]:
        """Get the assigned voice for a specific agent."""
        return voice_assignments.get(agent_id)
    
    def get_available_voices(self) -> Dict[str, List[str]]:
        """Get all available voices by gender."""
        return self.voices.copy()
