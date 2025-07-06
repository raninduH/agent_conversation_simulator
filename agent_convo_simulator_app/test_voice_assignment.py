#!/usr/bin/env python3
"""
Test script for voice assignment functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import Agent
from voice_assignment import VoiceAssignmentManager

def test_voice_assignment():
    """Test the voice assignment functionality."""
    print("Testing Voice Assignment Manager...")
    
    # Create some test agents
    agents = [
        Agent.create_new("Alice", "Teacher", "You are a helpful teacher.", gender="female"),
        Agent.create_new("Bob", "Student", "You are a curious student.", gender="male"),
        Agent.create_new("Charlie", "Moderator", "You moderate discussions.", gender="male"),
        Agent.create_new("Diana", "Expert", "You are a subject expert.", gender="female"),
    ]
    
    print(f"Created {len(agents)} test agents:")
    for agent in agents:
        print(f"  - {agent.name} ({agent.gender})")
    
    # Test voice assignment
    voice_manager = VoiceAssignmentManager()
    assignments = voice_manager.assign_voices_to_agents(agents)
    
    print("\nVoice assignments:")
    for agent in agents:
        voice = assignments.get(agent.id, "No voice assigned")
        print(f"  - {agent.name} ({agent.gender}): {voice}")
    
    # Test with existing assignments
    print("\nTesting with existing assignments...")
    existing = {agents[0].id: "af_bella"}  # Pre-assign Alice
    new_assignments = voice_manager.assign_voices_to_agents(agents, existing)
    
    print("Updated voice assignments:")
    for agent in agents:
        voice = new_assignments.get(agent.id, "No voice assigned")
        print(f"  - {agent.name} ({agent.gender}): {voice}")
    
    print("\nVoice assignment test completed successfully!")

if __name__ == "__main__":
    test_voice_assignment()
