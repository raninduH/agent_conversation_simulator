"""
Test script for the Multi-Agent Conversation Simulator
This script tests the basic functionality without requiring LangGraph dependencies.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_data_manager():
    """Test the data manager functionality."""
    print("Testing Data Manager...")
    
    try:
        from data_manager import DataManager, Agent, Conversation
        
        # Create a test data manager
        test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(test_dir, exist_ok=True)
        
        dm = DataManager(test_dir)
        
        # Test agent creation
        agent = Agent.create_new(
            name="Test Agent",
            role="Tester",
            base_prompt="You are a test agent for verification purposes.",
            personality_traits=["helpful", "thorough"]
        )
        
        # Test saving and loading agents
        dm.save_agent(agent)
        loaded_agents = dm.load_agents()
        
        assert len(loaded_agents) >= 1, "Agent not saved or loaded correctly"
        assert loaded_agents[0].name == "Test Agent", "Agent name not preserved"
        
        # Test conversation creation
        conversation = Conversation.create_new(
            title="Test Conversation",
            environment="Test Environment",
            scene_description="A test scene for verification",
            agent_ids=[agent.id]
        )
        
        # Test saving and loading conversations
        dm.save_conversation(conversation)
        loaded_conversations = dm.load_conversations()
        
        assert len(loaded_conversations) >= 1, "Conversation not saved or loaded correctly"
        assert loaded_conversations[0].title == "Test Conversation", "Conversation title not preserved"
        
        print("✅ Data Manager tests passed!")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Data Manager test failed: {e}")
        return False

def test_gui_imports():
    """Test GUI-related imports."""
    print("Testing GUI imports...")
    
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext
        
        # Test basic tkinter functionality
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test that we can create basic widgets
        label = ttk.Label(root, text="Test")
        entry = ttk.Entry(root)
        button = ttk.Button(root, text="Test")
        
        root.destroy()
        
        print("✅ GUI imports tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ GUI imports test failed: {e}")
        return False

def test_json_files():
    """Test JSON file structure."""
    print("Testing JSON files...")
    
    try:
        # Test agents.json
        agents_file = os.path.join(os.path.dirname(__file__), "agents.json")
        if os.path.exists(agents_file):
            with open(agents_file, 'r') as f:
                agents_data = json.load(f)
            assert "agents" in agents_data, "agents.json missing 'agents' key"
            print("✅ agents.json structure is valid")
        
        # Test conversations.json
        conversations_file = os.path.join(os.path.dirname(__file__), "conversations.json")
        if os.path.exists(conversations_file):
            with open(conversations_file, 'r') as f:
                conversations_data = json.load(f)
            assert "conversations" in conversations_data, "conversations.json missing 'conversations' key"
            print("✅ conversations.json structure is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON files test failed: {e}")
        return False

def test_main_imports():
    """Test main application imports (excluding LangGraph dependencies)."""
    print("Testing main application imports...")
    
    try:
        # Test standard library imports
        import threading
        import json
        import os
        from datetime import datetime
        from typing import List, Dict, Any, Optional
        
        print("✅ Standard library imports successful")
        
        # Test data manager import
        from data_manager import DataManager, Agent, Conversation
        print("✅ Data manager import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Main imports test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Multi-Agent Conversation Simulator Tests")
    print("=" * 50)
    
    tests = [
        test_gui_imports,
        test_main_imports,
        test_json_files,
        test_data_manager,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The basic application structure is working correctly.")
        print("\nTo run the full application:")
        print("1. Install LangGraph dependencies: pip install -r requirements.txt")
        print("2. Get a Google AI API key from https://console.cloud.google.com/")
        print("3. Run: python main.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
