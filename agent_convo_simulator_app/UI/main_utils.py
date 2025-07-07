import tkinter as tk
from tkinter import messagebox
import time

def _generate_clone_name(base_name: str, existing_agents: list) -> str:
    """Generate a unique clone name following the pattern {agent_name}_clone_{i}."""
    existing_names = {agent.name for agent in existing_agents}
    
    # Check for available clone number from 1 to 100
    for i in range(1, 101):
        clone_name = f"{base_name}_clone_{i}"
        if clone_name not in existing_names:
            return clone_name
    
    # If all clone names 1-100 are taken, fallback to timestamp-based naming
    timestamp = str(int(time.time()))[-4:]  # Last 4 digits of timestamp
    return f"{base_name}_clone_{timestamp}"

def _select_agent_by_name(app, agent_name: str):
    """Select an agent in the listbox by name."""
    try:
        for i in range(app.agents_listbox.size()):
            item_text = app.agents_listbox.get(i)
            if agent_name in item_text:
                app.agents_listbox.selection_clear(0, tk.END)
                app.agents_listbox.selection_set(i)
                app.agents_listbox.see(i)
                # Trigger the selection event to load agent details
                app.agents_listbox.event_generate('<<ListboxSelect>>')
                break
    except Exception as e:
        print(f"Error selecting agent: {e}")

def _toggle_termination_condition(app):
    """Toggle the termination condition text field and agent selector API key based on invocation method."""
    # Always ensure termination condition is enabled for both modes
    app.termination_condition_text.config(state=tk.NORMAL)
    
    # If Agent Selector is chosen, enable the API key field
    if app.invocation_method_var.get() == "agent_selector":
        app.agent_selector_api_key_entry.config(state=tk.NORMAL)
    else:
        # For Round Robin, disable agent selector API key
        # Clear and disable agent selector API key for round robin
        app.agent_selector_api_key_var.set("")
        app.agent_selector_api_key_entry.config(state=tk.DISABLED)
