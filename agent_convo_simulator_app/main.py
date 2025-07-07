"""
Multi-Agent Conversation Simulator GUI
A desktop application for simulating group conversations between AI agents using LangGraph and Gemini.
"""

import tkinter as tk
import os
import sys

# Add the parent directory to the Python path to allow for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .main_app import AgentConversationSimulatorGUI

if __name__ == "__main__":
    """Main entry point for the application."""
    root = tk.Tk()
    app = AgentConversationSimulatorGUI(root)
    root.mainloop()
