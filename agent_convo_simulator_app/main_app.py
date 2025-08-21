"""
Main application class for the Multi-Agent Conversation Simulator GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import threading
import json
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import importlib
import shutil
from tkinter import filedialog

# Import our custom modules
from agent_convo_simulator_app.data_manager import DataManager, Agent, Conversation

from agent_convo_simulator_app.conversation_engine import ConversationEngine
from agent_convo_simulator_app.config import UI_COLORS, AGENT_SETTINGS
from agent_convo_simulator_app import knowledge_manager # Import the new module
from agent_convo_simulator_app.audio_manager import AudioManager
from agent_convo_simulator_app.UI.chat_widgets import ChatCanvas
from agent_convo_simulator_app.UI.agent_management import AgentManagementTab
from agent_convo_simulator_app.UI.conversation_setup import ConversationSetupTab
from agent_convo_simulator_app.UI.simulation_tab import SimulationTab
from agent_convo_simulator_app.UI.past_conversations import PastConversationsTab
from agent_convo_simulator_app.UI.group_research_tab import GroupResearchTab
from agent_convo_simulator_app.UI.research_conversation_tab import ResearchConversationTab

class AgentConversationSimulatorGUI:
    """Main GUI application for the multi-agent conversation simulator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Agent Conversation Simulator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=UI_COLORS["app_background"])

        # Add a variable to hold the tooltip window
        self.tooltip_window = None
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Initialize audio manager
        try:
            self.audio_manager = AudioManager()
            self.audio_manager.set_audio_ready_callback(self.on_audio_ready)
            self.audio_manager.set_audio_finished_callback(self.on_audio_finished)
            self.audio_enabled = True
        except Exception as e:
            print(f"Warning: Could not initialize audio manager: {e}")
            self.audio_manager = None
            self.audio_enabled = False
        
        # Audio-related state
        self.pending_audio_messages = {}  # message_id -> message_data for messages waiting for audio
        self.audio_queue = []  # Queue of messages waiting to be sent to TTS
        self.current_playing_message_id = None
        
        # Message state tracking for pause cleanup
        self.displayed_messages = set()  # Set of message IDs that have been displayed
        self.message_bubbles = {}  # message_id -> bubble_widget mapping for removal
        self.removed_messages = set()  # Set of message IDs that were removed during pause cleanup
        
        # Initialize knowledge manager
        self.knowledge_files = {} # To store paths of files to be uploaded
        self.current_editing_agent_id = None  # Track which agent is being edited
        # Note: Embedding model will be loaded lazily when first needed

        # Initialize conversation engine (will be created when needed)
        self.conversation_engine = None
        self.current_conversation_id = None
        self.conversation_active = False
        
        # Initialize tool-related variables
        self.tooltip = None  # For displaying tool descriptions
        
        # Register window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create the main interface
        self.create_widgets()
        self.load_data()
          # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)


    def on_closing(self):
        """Handle application closing - stop any active conversations and clean up resources."""
        try:
            print("DEBUG: Application closing, cleaning up resources...")
            
            # Stop any active conversation
            if self.conversation_active and self.conversation_engine and self.current_conversation_id:
                print("DEBUG: Stopping active conversation before closing...")
                try:
                    # Try to pause the conversation first
                    self.conversation_engine.pause_conversation(self.current_conversation_id)
                    print("DEBUG: Conversation paused successfully")
                except Exception as e:
                    print(f"WARNING: Error pausing conversation: {e}")
                
                try:
                    # Try to gracefully stop the conversation engine
                    self.conversation_engine.stop_conversation(self.current_conversation_id)
                    print("DEBUG: Conversation stopped successfully")
                except Exception as e:
                    print(f"WARNING: Error stopping conversation: {e}")
            
            # Stop audio manager
            if hasattr(self, 'audio_manager'):
                try:
                    self.audio_manager.stop()
                    print("DEBUG: Audio manager stopped successfully")
                except Exception as e:
                    print(f"WARNING: Error stopping audio manager: {e}")
                    
                # Update conversation status in database to reflect it was stopped
                try:
                    conversation = self.data_manager.get_conversation_by_id(self.current_conversation_id)
                    if conversation:
                        conversation.status = "stopped"
                        self.data_manager.save_conversation(conversation)
                        print(f"DEBUG: Updated conversation {self.current_conversation_id} status to 'stopped'")
                except Exception as e:
                    print(f"WARNING: Error updating conversation status: {e}")
            
            # Clean up any other resources here
            
            # Signal all threads to stop if possible
            if hasattr(self, 'conversation_engine') and self.conversation_engine:
                # Set a flag to stop background threads if the engine has such a mechanism
                if hasattr(self.conversation_engine, 'running'):
                    self.conversation_engine.running = False
                    print("DEBUG: Set conversation engine running flag to False")
                
                # Wait a moment for threads to clean up
                time.sleep(0.5)
            
            print("DEBUG: Application shutdown complete")
            # Destroy the application
            self.root.destroy()
            
        except Exception as e:
            print(f"ERROR during application shutdown: {e}")
            # Make sure the application closes even if there's an error
            self.root.destroy()

    def create_widgets(self):
        """Create the main GUI widgets."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create tabs
        self.agent_management_tab = AgentManagementTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.agent_management_tab, text="Agents")

        self.conversation_setup_tab = ConversationSetupTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.conversation_setup_tab, text="Conversation Setup")

        self.past_conversations_tab = PastConversationsTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.past_conversations_tab, text="Past Conversations")

        self.simulation_tab = SimulationTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.simulation_tab, text="Simulation")

        # Add Group Research tab
        self.group_research_tab = GroupResearchTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.group_research_tab, text="Group Research")

        # Add Research Conversation tab
        self.research_conversation_tab = ResearchConversationTab(self.notebook, self, self.data_manager)
        self.notebook.add(self.research_conversation_tab, text="Research Conversation")
                
        # Status bar
        status_text = "Ready"
        self.status_bar = ttk.Label(self.root, text=status_text, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

    def show_tooltip(self, widget, text):
        """Display a tooltip for a given widget."""
        if self.tooltip_window:
            self.tooltip_window.destroy()

        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def load_data(self):
        """Load agents and conversations from JSON files."""
        self.agent_management_tab.refresh_agents_list()
        self.conversation_setup_tab.refresh_agent_checkboxes()
        self.past_conversations_tab.refresh_past_conversations()

    def update_status(self, message: str):
        """Update the status message (placeholder for now)."""
        print(f"STATUS: {message}")
        self.status_bar.config(text=message)

    def pause_conversation(self):
        """Pauses the current conversation from the UI."""
        print("[MainApp] pause_conversation called")
        if self.conversation_active and self.conversation_engine and self.current_conversation_id:
            print("[MainApp] Calling engine.pause_conversation...")
            self.conversation_engine.pause_conversation(self.current_conversation_id)
            print("[MainApp] Calling simulation_tab.pause_conversation...")
            self.simulation_tab.pause_conversation()
            print("[MainApp] Called simulation_tab.pause_conversation")
            self.update_status("Conversation paused.")
        else:
            print("[MainApp] pause_conversation called but not active or engine missing")

    def resume_conversation(self):
        """Resumes the current conversation from the UI."""
        print("[MainApp] resume_conversation called")
        if self.conversation_active and self.conversation_engine and self.current_conversation_id:
            print("[MainApp] Calling engine.resume_conversation...")
            self.conversation_engine.resume_conversation(self.current_conversation_id)
            print("[MainApp] Calling simulation_tab.resume_conversation...")
            self.simulation_tab.resume_conversation()
            print("[MainApp] Called simulation_tab.resume_conversation")
            self.update_status("Conversation resumed.")
        else:
            print("[MainApp] resume_conversation called but not active or engine missing")

    def stop_conversation_ui(self):
        """Stops the current conversation from the UI."""
        if self.conversation_active and self.conversation_engine and self.current_conversation_id:
            self.conversation_engine.stop_conversation(self.current_conversation_id)
            self.conversation_active = False
            self.simulation_tab.update_simulation_controls(False)
            self.update_status("Conversation stopped by user.")
            # Update conversation status in the database
            try:
                conversation = self.data_manager.get_conversation_by_id(self.current_conversation_id)
                if conversation:
                    conversation.status = "stopped"
                    self.data_manager.save_conversation(conversation)
            except Exception as e:
                print(f"Error updating conversation status: {e}")

    def summarize_conversation(self):
        """Summarizes the current conversation."""
        if not self.conversation_active or not self.conversation_engine or not self.current_conversation_id:
            messagebox.showwarning("Warning", "No active conversation to summarize.")
            return
        
        try:
            self.update_status("Generating conversation summary...")
            
            # Get summary from the conversation engine
            summary = self.conversation_engine.get_conversation_summary(self.current_conversation_id)
            
            if summary:
                # Create a dialog window to display the summary
                summary_window = tk.Toplevel(self.root)
                summary_window.title("Conversation Summary")
                summary_window.geometry("600x400")
                summary_window.configure(bg=UI_COLORS["app_background"])
                
                # Make it modal
                summary_window.transient(self.root)
                summary_window.grab_set()
                
                # Center the window
                summary_window.geometry("+{}+{}".format(
                    int(self.root.winfo_x() + (self.root.winfo_width() / 2) - 300),
                    int(self.root.winfo_y() + (self.root.winfo_height() / 2) - 200)
                ))
                
                # Title label
                title_label = ttk.Label(summary_window, text="Conversation Summary", 
                                      font=("Arial", 14, "bold"))
                title_label.pack(pady=10)
                
                # Text widget with scrollbar for the summary
                text_frame = ttk.Frame(summary_window)
                text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
                
                summary_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                       font=("Arial", 10),
                                                       bg="white", fg="black")
                summary_text.pack(fill=tk.BOTH, expand=True)
                
                # Insert the summary text
                summary_text.insert(tk.END, summary)
                summary_text.config(state=tk.DISABLED)  # Make it read-only
                
                # Close button
                button_frame = ttk.Frame(summary_window)
                button_frame.pack(pady=(0, 10))
                
                close_btn = ttk.Button(button_frame, text="Close", 
                                     command=summary_window.destroy)
                close_btn.pack()
                
                self.update_status("Summary generated successfully.")
            else:
                messagebox.showerror("Error", "Failed to generate conversation summary.")
                self.update_status("Summary generation failed.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating summary: {str(e)}")
            self.update_status("Summary generation failed.")

    def send_user_message(self, event=None):
        """Send a user message to the conversation."""
        if not self.conversation_active or not self.conversation_engine or not self.current_conversation_id:
            return
        # Get message from the entry field
        message = getattr(self, 'message_var', tk.StringVar()).get().strip()
        if not message:
            return
        # Clear the entry field
        self.message_var.set("")
        # Display the user message in the chat
        user_message_data = {
            "sender": "You",
            "content": message,
            "type": "user",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.simulation_tab.display_message(user_message_data)
        # Send message to the backend (RoundRobinEngine)
        self.conversation_engine.on_user_message(self.current_conversation_id, user_message_data)

    def change_scene(self):
        """Change the environment and scene during a conversation."""
        if not self.conversation_active or not self.conversation_engine or not self.current_conversation_id:
            messagebox.showwarning("Warning", "No active conversation to change.")
            return
        
        new_env = getattr(self, 'new_env_var', tk.StringVar()).get().strip()
        new_scene = getattr(self, 'new_scene_text', None)
        new_scene_text = new_scene.get(1.0, tk.END).strip() if new_scene else ""
        
        if not new_env and not new_scene_text:
            messagebox.showwarning("No Change", "Please provide a new environment or scene.")
            return
        
        try:
            # Create a scene change message for the conversation
            scene_change_message = f"SCENE CHANGE: "
            if new_env:
                scene_change_message += f"Environment changed to: {new_env}. "
            if new_scene_text:
                scene_change_message += f"Scene description: {new_scene_text}"
            
            # Send the scene change as a system message
            threading.Thread(
                target=self._send_message_thread,
                args=(scene_change_message, "system"),
                daemon=True
            ).start()
            
            # Update the UI
            if new_env and hasattr(self, 'current_env_label'):
                self.current_env_label.config(text=new_env)
                self.new_env_var.set("")
            if new_scene_text and hasattr(self, 'new_scene_text'):
                self.new_scene_text.delete(1.0, tk.END)
                
            self.update_status("Scene updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change scene: {str(e)}")

    def start_conversation(self):
        """Start a new conversation."""
        print(f"[MainApp] Entered Start conversation method")
        title = self.conv_title_var.get().strip()
        environment = self.conv_env_var.get().strip()
        scene = self.conv_scene_text.get(1.0, tk.END).strip()
        if not all([title, environment, scene]):
            messagebox.showwarning("Missing Info", "Please fill in all conversation details.")
            return
        selected_agents = self.conversation_setup_tab.update_selected_agents()
        if len(selected_agents) < 2 or len(selected_agents) > 4:
            messagebox.showwarning("Agent Selection", "Please select 2-4 agents.")
            return
        # Assign unique colors to agents
        from agent_convo_simulator_app.config import UI_COLORS
        available_colors = UI_COLORS["agent_colors"][:]
        random.shuffle(available_colors)
        agent_colors = {}
        for i, agent_id in enumerate(selected_agents):
            agent_colors[agent_id] = available_colors[i % len(available_colors)]
        try:
            conversation = Conversation.create_new(
                title=title,
                environment=environment,
                scene_description=scene,
                agent_ids=selected_agents,
                invocation_method=self.invocation_method_var.get(),
                termination_condition=self.termination_condition_text.get(1.0, tk.END).strip(),
                agent_selector_api_key=self.agent_selector_api_key_var.get().strip(),
                voices_enabled=self.voices_enabled_var.get()
            )
            conversation.agent_colors = agent_colors
            # Ensure agent_numbers is present and correct
            conversation.agent_numbers = {agent_id: i+1 for i, agent_id in enumerate(selected_agents)}
            self.data_manager.save_conversation(conversation)
            self.current_conversation_id = conversation.id
            self.conversation_active = True
            # Register message callback for the correct engine
            if hasattr(self.conversation_engine, 'register_message_callback'):
                print(f"[MainApp] Registering message callback for '{conversation.id}' with invocation method '{conversation.invocation_method}'")
                self.conversation_engine.register_message_callback(conversation.id, self.simulation_tab.handle_message_callback)
           

                
            self.simulation_tab.load_conversation(conversation)
            # After starting, switch to Simulation tab
            self.notebook.select(self.simulation_tab)
            self.update_status("Conversation started. Switched to Simulation tab.")

            try:
                self.conversation_engine.resume_conversation(conversation.id)
            except Exception as e:
                print(f"[MainApp] Error in start_conversation: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start conversation: {str(e)}")
            self.update_status("Failed to start conversation.")

    def send_initial_message(self):
        """Send an initial message to start the conversation."""
        if self.conversation_engine and self.current_conversation_id:
            threading.Thread(
                target=self._send_message_thread,
                args=("Hello everyone! Let's start our conversation.", "system"),
                daemon=True
            ).start()
    
    def _send_message_thread(self, message: str, sender: str = "user"):
        """Send message in a separate thread."""
        try:
            response = self.conversation_engine.send_message(
                self.current_conversation_id, message, sender
            )
            
            if not response.get("success"):
                self.root.after(0, lambda: self.update_status(f"Error: {response.get('error', 'Unknown error')}"))
                
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error sending message: {str(e)}"))

    def load_selected_conversation(self, conversation):
        """Loads a selected conversation into the simulation tab and resumes it."""
        try:
            # Resume the conversation using the new ConversationEngine logic
            if not self.conversation_engine:
                self.conversation_engine = ConversationEngine()
            # Register message callback for the correct engine
            if hasattr(self.conversation_engine, 'register_message_callback'):
                self.conversation_engine.register_message_callback(conversation.id, self.simulation_tab.handle_message_callback)
            self.conversation_engine.resume_conversation(conversation.id)
            self.current_conversation_id = conversation.id
            self.conversation_active = True
            # Update the simulation tab UI
            self.simulation_tab.load_conversation(conversation)
            # Switch to Simulation tab after loading a conversation
            self.notebook.select(self.simulation_tab)
            self.update_status(f"Loaded conversation: {conversation.title}. Switched to Simulation tab.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversation: {str(e)}")
            self.update_status(f"Failed to load conversation: {str(e)}")

    def on_message_received(self, message_data: Dict[str, Any]):
        """Callback for when a new message is received."""
        if not self.conversation_active:
            print("[ERROR] Conversation not active. Message not displayed.")
            return
        try:
            self.simulation_tab.display_message(message_data)
        except Exception as e:
            print(f"[ERROR] Failed to display chat bubble: {e}")
            import traceback
            traceback.print_exc()

    def on_audio_ready(self, conv_id, agent_id, message_id):
        """Callback for when audio is ready."""
        if self.conversation_active:
            self.simulation_tab.on_audio_ready(conv_id, agent_id, message_id)

    def on_audio_finished(self, conv_id, agent_id, message_id):
        """Callback for when audio has finished playing."""
        if self.conversation_active:
            self.simulation_tab.on_audio_finished(conv_id, agent_id, message_id)
