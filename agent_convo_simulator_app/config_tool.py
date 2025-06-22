#!/usr/bin/env python3
"""
Configuration utility for the Agent Conversation Simulator.
This script allows you to easily view and modify configuration settings.
"""

import json
from typing import Any, Dict
from config import CONVERSATION_TIMING, MESSAGE_SETTINGS, AGENT_SETTINGS, MODEL_SETTINGS


def display_current_config():
    """Display the current configuration settings."""
    print("=" * 60)
    print("CURRENT CONFIGURATION SETTINGS")
    print("=" * 60)
    
    print("\n📅 CONVERSATION TIMING SETTINGS:")
    print(f"  • Start delay: {CONVERSATION_TIMING['start_delay']} seconds")
    print(f"  • Agent turn delay: {CONVERSATION_TIMING['agent_turn_delay_min']}-{CONVERSATION_TIMING['agent_turn_delay_max']} seconds")
    print(f"  • Resume delay: {CONVERSATION_TIMING['resume_delay']} seconds")
    print(f"  • Error retry delay: {CONVERSATION_TIMING['error_retry_delay']} seconds")
    
    print("\n💬 MESSAGE SETTINGS:")
    print(f"  • Max messages before summary: {MESSAGE_SETTINGS['max_messages_before_summary']}")
    print(f"  • Messages to keep after summary: {MESSAGE_SETTINGS['messages_to_keep_after_summary']}")
    
    print("\n🤖 AGENT SETTINGS:")
    print(f"  • Response temperature: {AGENT_SETTINGS['response_temperature']}")
    print(f"  • Summary temperature: {AGENT_SETTINGS['summary_temperature']}")
    print(f"  • Max retries: {AGENT_SETTINGS['max_retries']}")
    
    print("\n🧠 MODEL SETTINGS:")
    print(f"  • Agent model: {MODEL_SETTINGS['agent_model']}")
    print(f"  • Summary model: {MODEL_SETTINGS['summary_model']}")
    print()


def update_agent_turn_delay():
    """Update the agent turn delay settings."""
    print("\n🔧 UPDATING AGENT TURN DELAY")
    print("Current settings:")
    print(f"  Min delay: {CONVERSATION_TIMING['agent_turn_delay_min']} seconds")
    print(f"  Max delay: {CONVERSATION_TIMING['agent_turn_delay_max']} seconds")
    
    try:
        min_delay = float(input("\nEnter new minimum delay (seconds): "))
        max_delay = float(input("Enter new maximum delay (seconds): "))
        
        if min_delay >= max_delay:
            print("❌ Error: Minimum delay must be less than maximum delay!")
            return False
        
        if min_delay < 0 or max_delay < 0:
            print("❌ Error: Delays must be positive numbers!")
            return False
        
        # Update config.py file
        update_config_file('agent_turn_delay_min', min_delay)
        update_config_file('agent_turn_delay_max', max_delay)
        
        print(f"✅ Updated agent turn delay to {min_delay}-{max_delay} seconds")
        print("⚠️  Note: Restart the application for changes to take effect.")
        return True
        
    except ValueError:
        print("❌ Error: Please enter valid numbers!")
        return False


def update_config_file(key: str, value: Any):
    """Update a specific key in the config.py file."""
    # Read the current config file
    with open('config.py', 'r') as f:
        content = f.read()
    
    # This is a simple approach - for a more robust solution, 
    # you might want to use AST manipulation
    if key in ['agent_turn_delay_min', 'agent_turn_delay_max', 'start_delay', 'resume_delay', 'error_retry_delay']:
        # Update timing settings
        old_line = f'    "{key}": {CONVERSATION_TIMING[key]}'
        new_line = f'    "{key}": {value}'
        content = content.replace(old_line, new_line)
    
    # Write back to file
    with open('config.py', 'w') as f:
        f.write(content)


def interactive_config():
    """Interactive configuration menu."""
    while True:
        print("\n" + "=" * 60)
        print("AGENT CONVERSATION SIMULATOR - CONFIGURATION")
        print("=" * 60)
        print("1. View current configuration")
        print("2. Update agent turn delay")
        print("3. Quick timing presets")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            display_current_config()
        elif choice == '2':
            update_agent_turn_delay()
        elif choice == '3':
            show_timing_presets()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


def show_timing_presets():
    """Show and apply timing presets."""
    print("\n🚀 TIMING PRESETS")
    print("1. Fast (0.5-1.5 seconds) - Quick conversations")
    print("2. Normal (2-4 seconds) - Default balanced timing")
    print("3. Slow (5-8 seconds) - Thoughtful conversations")
    print("4. Very Slow (10-15 seconds) - Deliberate pacing")
    print("5. Custom - Set your own values")
    
    choice = input("\nSelect a preset (1-5): ").strip()
    
    presets = {
        '1': (0.5, 1.5, "Fast"),
        '2': (2.0, 4.0, "Normal"),
        '3': (5.0, 8.0, "Slow"),
        '4': (10.0, 15.0, "Very Slow")
    }
    
    if choice in presets:
        min_delay, max_delay, name = presets[choice]
        update_config_file('agent_turn_delay_min', min_delay)
        update_config_file('agent_turn_delay_max', max_delay)
        print(f"✅ Applied {name} preset: {min_delay}-{max_delay} seconds")
        print("⚠️  Note: Restart the application for changes to take effect.")
    elif choice == '5':
        update_agent_turn_delay()
    else:
        print("❌ Invalid choice.")


if __name__ == "__main__":
    print("🎛️  Agent Conversation Simulator - Configuration Tool")
    interactive_config()
