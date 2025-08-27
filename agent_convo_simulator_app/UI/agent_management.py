import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog, filedialog
import os
import shutil
from datetime import datetime
import time

from ..data_manager import Agent
from ..knowledge_manager import knowledge_manager
from .main_utils import _generate_clone_name, _select_agent_by_name

class AgentManagementTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager

        # Initialize tool-related variables
        self.tool_vars = {}  # For tool checkboxes
        self.tools_checkboxes_frame = None  # Will be set in create_agents_tab
        self.tooltip = None  # For displaying tool descriptions

        # Initialize knowledge manager
        self.knowledge_files = {} # To store paths of files to be uploaded
        self.current_editing_agent_id = None  # Track which agent is being edited

        self.create_widgets()

    def create_widgets(self):
        """Create the agents management tab."""
        agents_frame = self
        
        # Configure grid
        agents_frame.grid_rowconfigure(1, weight=1)
        agents_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(agents_frame, text="Agent Management", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))
        
        # Left panel - Agent list
        list_frame = ttk.LabelFrame(agents_frame, text="Agents", padding="10")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Agent listbox with scrollbar
        list_container = tk.Frame(list_frame)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)
        
        self.app.agents_listbox = tk.Listbox(list_container)
        self.app.agents_listbox.grid(row=0, column=0, sticky="nsew")
        self.app.agents_listbox.bind('<<ListboxSelect>>', self.on_agent_select)
        
        agents_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.app.agents_listbox.yview)
        agents_scrollbar.grid(row=0, column=1, sticky="ns")
        self.app.agents_listbox.configure(yscrollcommand=agents_scrollbar.set)
        
        # Buttons for agent management
        btn_frame = ttk.Frame(list_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(btn_frame, text="New Agent", command=self.new_agent).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clone Agent", command=self.clone_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Agent", command=self.delete_agent).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Agent details
        details_frame = ttk.LabelFrame(agents_frame, text="Agent Details", padding="10")
        details_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        details_frame.grid_columnconfigure(1, weight=1)
        
        # Agent details form
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.app.agent_name_var = tk.StringVar()
        self.app.agent_name_entry = ttk.Entry(details_frame, textvariable=self.app.agent_name_var, width=30)
        self.app.agent_name_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(details_frame, text="Role:").grid(row=1, column=0, sticky="w", pady=2)
        self.app.agent_role_var = tk.StringVar()
        self.app.agent_role_entry = ttk.Entry(details_frame, textvariable=self.app.agent_role_var, width=30)
        self.app.agent_role_entry.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(details_frame, text="Gender:").grid(row=2, column=0, sticky="w", pady=2)
        self.app.agent_gender_var = tk.StringVar()
        self.app.agent_gender_combo = ttk.Combobox(details_frame, textvariable=self.app.agent_gender_var, width=27, state="readonly")
        self.app.agent_gender_combo['values'] = ("male", "female")
        self.app.agent_gender_combo.grid(row=2, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        # Voice selection (move to row 3)
        ttk.Label(details_frame, text="Voice:").grid(row=3, column=0, sticky="w", pady=2)
        self.app.agent_voice_var = tk.StringVar()
        self.app.agent_voice_combo = ttk.Combobox(details_frame, textvariable=self.app.agent_voice_var, width=27, state="readonly")
        self.app.agent_voice_combo.grid(row=3, column=1, sticky="ew", pady=2, padx=(10, 0))
        self.app.agent_voice_combo['values'] = ()  # Will be set based on gender
        self.app.agent_voice_combo.set('')
        self.app.play_voice_btn = ttk.Button(details_frame, text="Play Sample", command=self.play_selected_voice_sample)
        self.app.play_voice_btn.grid(row=3, column=2, padx=(5, 0))
        self.app.play_voice_btn['state'] = 'disabled'
        self.app.agent_gender_var.trace_add('write', lambda *args: self.update_voice_options())
        self.app.agent_voice_var.trace_add('write', lambda *args: self.update_play_button_state())
        
        # Personality Traits (move to row 4)
        ttk.Label(details_frame, text="Personality Traits:").grid(row=4, column=0, sticky="nw", pady=2)
        self.app.agent_traits_var = tk.StringVar()
        self.app.agent_traits_entry = ttk.Entry(details_frame, textvariable=self.app.agent_traits_var, width=30)
        self.app.agent_traits_entry.grid(row=4, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        # Base Prompt (move to row 5)
        ttk.Label(details_frame, text="Base Prompt:").grid(row=5, column=0, sticky="nw", pady=2)
        self.app.agent_prompt_text = scrolledtext.ScrolledText(details_frame, width=40, height=10)
        self.app.agent_prompt_text.grid(row=5, column=1, sticky="nsew", pady=2, padx=(10, 0))
        details_frame.grid_rowconfigure(5, weight=1)
        
        ttk.Label(details_frame, text="API Key:").grid(row=6, column=0, sticky="w", pady=2)
        self.app.agent_api_key_var = tk.StringVar()
        self.app.agent_api_key_entry = ttk.Entry(details_frame, textvariable=self.app.agent_api_key_var, width=30, show="*")
        self.app.agent_api_key_entry.grid(row=6, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        # Tools selection
        ttk.Label(details_frame, text="Tools:").grid(row=7, column=0, sticky="nw", pady=2)
        
        # Create a frame for tools with a scrollbar
        tools_frame = ttk.Frame(details_frame)
        tools_frame.grid(row=7, column=1, sticky="nsew", pady=2, padx=(10, 0))
        details_frame.grid_rowconfigure(7, weight=1)
        
        # Create a canvas for scrolling
        tools_canvas = tk.Canvas(tools_frame, height=100)
        tools_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        tools_scrollbar = ttk.Scrollbar(tools_frame, orient="vertical", command=tools_canvas.yview)
        tools_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tools_canvas.configure(yscrollcommand=tools_scrollbar.set)
        
        # Create a frame inside the canvas to hold the checkboxes
        self.tools_checkboxes_frame = ttk.Frame(tools_canvas)
        tools_canvas.create_window((0, 0), window=self.tools_checkboxes_frame, anchor="nw")
        
        # Configure the canvas to scroll with the content
        self.tools_checkboxes_frame.bind("<Configure>", 
            lambda e: tools_canvas.configure(scrollregion=tools_canvas.bbox("all")))
        
        # Dictionary to hold the checkbox variables
        self.tool_vars = {}
        
        # Load tools and create checkboxes
        self.load_tool_checkboxes()

        # Knowledge Base Section
        kb_frame = ttk.LabelFrame(details_frame, text="Knowledge Base", padding="5")
        kb_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=(0,0))
        kb_frame.grid_columnconfigure(1, weight=1)


        self.app.upload_kb_btn = ttk.Button(kb_frame, text="Upload Files (.pdf, .txt)", command=self.upload_knowledge_files, state=tk.NORMAL)
        self.app.upload_kb_btn.grid(row=0, column=0, padx=5, pady=5)

        # Add Show Existing Knowledge button
        self.app.show_existing_kb_btn = ttk.Button(kb_frame, text="Show Existing Knowledge", command=self.show_existing_knowledge)
        self.app.show_existing_kb_btn.grid(row=0, column=2, padx=5, pady=5)

        self.app.knowledge_files_label = ttk.Label(kb_frame, text="No files selected.")
        self.app.knowledge_files_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Save button
        ttk.Button(details_frame, text="Save Agent", command=self.save_agent).grid(row=9, column=1, sticky="e", pady=(10, 0))

    def show_existing_knowledge(self):
        """Show a popup listing the current agent's knowledge base documents and descriptions."""
        # Determine which agent is currently being edited/selected
        agent = None
        if self.current_editing_agent_id:
            agent = self.data_manager.get_agent_by_id(self.current_editing_agent_id)
        else:
            # Try to get agent by name if possible (for new agent, nothing to show)
            agent_name = self.app.agent_name_var.get().strip()
            if agent_name:
                agents = self.data_manager.load_agents()
                agent = next((a for a in agents if a.name == agent_name), None)
        if not agent or not hasattr(agent, 'knowledge_base') or not agent.knowledge_base:
            messagebox.showinfo("No Knowledge Base", "This agent has no knowledge base documents.")
            return

        popup = tk.Toplevel(self)
        popup.title("Existing Knowledge Base Documents")
        popup.geometry("600x350")
        popup.grab_set()
        label = tk.Label(popup, text="Current Knowledge Base Documents:", font=("Arial", 12, "bold"))
        label.pack(pady=(10, 5))

        # Use a Text widget for scrollable, formatted display
        text = tk.Text(popup, wrap="word", height=14, width=70)
        text.pack(padx=10, pady=5, fill="both", expand=True)
        text.config(state="normal")

        # Format and insert the knowledge base info
        for idx, doc in enumerate(agent.knowledge_base, 1):
            doc_name = doc.get("doc_name", "(No name)")
            desc = doc.get("description", "(No description)")
            text.insert("end", f"{idx}. {doc_name}\n   Description: {desc}\n\n")
        text.config(state="disabled")

        close_btn = ttk.Button(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=(0, 10))

    def on_agent_select(self, event):
        """Handle agent selection in the listbox."""
        selection = self.app.agents_listbox.curselection()
        if selection:
            agents = self.data_manager.load_agents()  # Use cached agents for selection
            if selection[0] < len(agents):
                agent = agents[selection[0]]
                self.load_agent_details(agent)

    def load_agent_details(self, agent: Agent):
        """Load agent details into the form."""
        self.current_editing_agent_id = agent.id  # Track that we're editing this agent
        self.app.agent_name_var.set(agent.name)
        self.app.agent_role_var.set(agent.role)
        self.app.agent_gender_var.set(getattr(agent, 'gender', 'Unspecified'))  # Load gender with default
        self.app.agent_traits_var.set(", ".join(agent.personality_traits))
        self.app.agent_api_key_var.set(agent.api_key or "")  # Load API key

        self.app.agent_prompt_text.delete(1.0, tk.END)
        self.app.agent_prompt_text.insert(1.0, agent.base_prompt)

        # Set tool checkboxes (excluding auto-managed tools)
        if hasattr(agent, 'tools') and self.tool_vars:
            for tool_name, var in self.tool_vars.items():
                # Only set checkbox for user-selectable tools (knowledge_base_retriever is auto-managed)
                if tool_name != 'knowledge_base_retriever':
                    var.set(tool_name in getattr(agent, 'tools', []))

        # Enable KB upload button and clear old file selections
        self.app.upload_kb_btn.config(state=tk.NORMAL)
        self.app.knowledge_files_label.config(text="No files selected.")
        self.knowledge_files = {}

        # Set voice selection and update combobox
        selected_voice = getattr(agent, 'voice', '')
        self.update_voice_options()  # Update options based on gender
        if selected_voice:
            self.app.agent_voice_var.set(selected_voice)
            self.app.agent_voice_combo.set(selected_voice)
        else:
            self.app.agent_voice_var.set('')
            self.app.agent_voice_combo.set('')
        self.app.play_voice_btn.config(state=tk.NORMAL if selected_voice else tk.DISABLED)

    def new_agent(self):
        """Create a new agent."""
        self.clear_agent_form()
        self.current_editing_agent_id = None  # Clear editing state for new agent
        self.app.agent_name_entry.focus()
        # Enable upload button for new agents
        self.app.upload_kb_btn.config(state=tk.NORMAL)
    
    def clear_agent_form(self):
        """Clear the agent form."""
        self.current_editing_agent_id = None  # Clear editing state
        self.app.agent_name_var.set("")
        self.app.agent_role_var.set("")
        self.app.agent_gender_var.set("")  # Clear gender
        self.app.agent_traits_var.set("")
        self.app.agent_api_key_var.set("")  # Clear API key
        self.app.agent_prompt_text.delete(1.0, tk.END)
        
        # Clear tool checkboxes
        for var in self.tool_vars.values():
            var.set(False)

        # Keep KB upload button enabled for new agent creation
        self.app.upload_kb_btn.config(state=tk.NORMAL)
        self.app.knowledge_files_label.config(text="No files selected.")
        # Don't clear knowledge_files completely - just clear any temporary staging
        temp_keys = [k for k in self.knowledge_files.keys() if k.startswith("NEW_AGENT_")]
        for temp_key in temp_keys:
            del self.knowledge_files[temp_key]

        # Clear voice selection
        self.app.agent_voice_var.set('')
        self.app.play_voice_btn.config(state=tk.DISABLED)

    def clone_agent(self):
        """Clone the selected agent with a unique name."""
        selection = self.app.agents_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an agent to clone.")
            return
        
        agents = self.data_manager.load_agents()  # Use cached agents for cloning
        if selection[0] < len(agents):
            original_agent = agents[selection[0]]
            
            # Generate unique clone name
            base_name = original_agent.name
            clone_name = _generate_clone_name(base_name, agents)
            
            # Create cloned agent
            cloned_agent = Agent.create_new(
                name=clone_name,
                role=original_agent.role,
                base_prompt=original_agent.base_prompt,
                personality_traits=original_agent.personality_traits.copy(),
                color=original_agent.color,
                api_key=original_agent.api_key,
                tools=original_agent.tools.copy(),
                gender=getattr(original_agent, 'gender', 'Unspecified')  # Copy gender with default
            )
            
            # Copy knowledge base if it exists
            if hasattr(original_agent, 'knowledge_base') and original_agent.knowledge_base:
                cloned_agent.knowledge_base = [doc.copy() for doc in original_agent.knowledge_base]
            
            # Auto-manage knowledge_base_retriever tool for the cloned agent
            self._update_knowledge_base_tool(cloned_agent)
            
            # Save the cloned agent
            self.data_manager.save_agent(cloned_agent)
            
            # Refresh UI
            self.refresh_agents_list()
            self.app.conversation_setup_tab.refresh_agent_checkboxes()
            
            # Select the newly cloned agent in the list
            _select_agent_by_name(self.app, clone_name)
            
            self.app.update_status(f"Agent '{clone_name}' cloned from '{original_agent.name}'.")

    def delete_agent(self):
        """Delete the selected agent."""
        selection = self.app.agents_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an agent to delete.")
            return
        
        agents = self.data_manager.load_agents()  # Use cached agents for delete
        if selection[0] < len(agents):
            agent = agents[selection[0]]
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete agent '{agent.name}'?"):
                self.data_manager.delete_agent(agent.id)
                self.refresh_agents_list()
                self.app.conversation_setup_tab.refresh_agent_checkboxes()
                self.clear_agent_form()
                self.app.update_status(f"Agent '{agent.name}' deleted.")

    def upload_knowledge_files(self):
        """Handle uploading knowledge base files for the selected agent."""
        print("\n" + "="*60)
        print("📁 KNOWLEDGE BASE FILE UPLOAD STARTED")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Check if we're editing an existing agent or creating a new one
        selection = self.app.agents_listbox.curselection()
        agent_name = self.app.agent_name_var.get().strip()
        
        if selection and self.current_editing_agent_id:
            # Editing existing agent - use the tracked agent ID
            agent_id = self.current_editing_agent_id
            agents = self.data_manager.load_agents()  # Use cached agents for upload
            agent = next((a for a in agents if a.id == agent_id), None)
            if agent:
                agent_display_name = agent.name
                print(f"✅ Editing existing agent:")
            else:
                print(f"❌ UPLOAD FAILED: Could not find agent with ID {agent_id}")
                messagebox.showerror("Agent Not Found", "Could not find the selected agent.")
                return
        else:
            # Creating new agent - use a temporary ID based on current form data
            if not agent_name:
                print("❌ UPLOAD FAILED: No agent name provided")
                print("💡 Please enter an agent name before uploading knowledge files")
                messagebox.showwarning("No Agent Name", "Please enter an agent name before uploading knowledge files.")
                return
            
            # Use a special temporary ID for new agents
            agent_id = f"NEW_AGENT_{agent_name.replace(' ', '_')}"
            agent_display_name = agent_name
            print(f"✅ Preparing files for new agent:")
        
        print(f"   🆔 Agent ID: {agent_id}")
        print(f"   👤 Agent Name: {agent_display_name}")

        print(f"\n📂 Opening file dialog...")
        file_paths = filedialog.askopenfilenames(
            title=f"Select Knowledge Files for {agent_display_name}",
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )

        if not file_paths:
            print("❌ UPLOAD CANCELLED: No files selected")
            print("="*60)
            return
            
        print(f"✅ Files selected: {len(file_paths)}")
        for i, file_path in enumerate(file_paths, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"   {i}. {file_name} ({file_size:,} bytes)")

        print(f"\n💬 Collecting document descriptions...")
        # Ask for descriptions for each file
        file_descriptions = {}
        for i, file_path in enumerate(file_paths, 1):
            file_name = os.path.basename(file_path)
            print(f"   📝 Requesting description for file {i}/{len(file_paths)}: {file_name}")
            
            description = simpledialog.askstring(
                "Document Description",
                f"Please provide a brief description of what '{file_name}' contains:",
                initialvalue=""
            )
            
            if description and description.strip():
                file_descriptions[file_path] = description.strip()
                print(f"      ✅ Description provided: '{description.strip()}'")
            else:
                default_desc = f"Document: {file_name}"
                file_descriptions[file_path] = default_desc.strip()
                print(f"      ⚠️  No description provided, using default: '{default_desc}'")
        
        print(f"\n📋 Staging files for agent {agent_id}...")
        # Store both file paths and descriptions
        self.knowledge_files[agent_id] = {
            'file_paths': file_paths,
            'descriptions': file_descriptions
        }
        
        print(f"✅ FILES STAGED SUCCESSFULLY!")
        print(f"   🎯 Agent: {agent_display_name} ({agent_id})")
        print(f"   📁 Files staged: {len(file_paths)}")
        print(f"   💬 Descriptions collected: {len(file_descriptions)}")
        print(f"   ⏳ Files will be processed when agent is saved")
        
        self.app.knowledge_files_label.config(text=f"{len(file_paths)} file(s) selected for upload.")
        print("="*60)

    def save_agent(self):
        """Save the current agent."""
        name = self.app.agent_name_var.get().strip()
        role = self.app.agent_role_var.get().strip()
        gender = self.app.agent_gender_var.get().strip()
        traits_str = self.app.agent_traits_var.get().strip()
        api_key = self.app.agent_api_key_var.get().strip()  # Get API key
        prompt = self.app.agent_prompt_text.get(1.0, tk.END).strip()
        voice = self.app.agent_voice_var.get().strip()  # Get voice
        if not all([name, role, prompt]):
            messagebox.showwarning("Missing Info", "Please fill in all required agent details.")
            return
        print("\n================ SAVE_AGENT CALLED ================")
        print(f"Agent name: {name}, role: {role}, gender: {gender}, voice: {voice}")
        print(f"Current editing agent id: {self.current_editing_agent_id}")
        # Parse personality traits
        traits = [t.strip() for t in traits_str.split(",") if t.strip()] if traits_str else []
        # Get selected tools (excluding knowledge_base_retriever as it's auto-managed)
        selected_tools = [name for name, var in self.tool_vars.items() if var.get()]
        # Track the agent_id for knowledge ingestion
        agent_id_for_ingestion = None
        # Check if knowledge_base_retriever should be added
        knowledge_base_nonempty = False
        staged_kb = []
        if self.current_editing_agent_id:
            # Update existing agent
            agent = self.data_manager.get_agent_by_id(self.current_editing_agent_id)
            if agent:
                print(f"Loaded existing agent: {agent.name} (id: {agent.id})")
                # If there are staged files, prepare staged_kb
                if self.current_editing_agent_id in self.knowledge_files:
                    file_data = self.knowledge_files[self.current_editing_agent_id]
                    file_paths = file_data['file_paths']
                    descriptions = file_data['descriptions']
                    print(f"Found staged files for agent {self.current_editing_agent_id}: {file_paths}")
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        description = descriptions.get(file_path, f"Document: {file_name}")
                        staged_kb.append({"doc_name": file_name, "description": description})
                    print(f"Staged KB to append: {staged_kb}")
                # Check if agent.knowledge_base is non-empty or staged_kb is non-empty
                if (hasattr(agent, 'knowledge_base') and agent.knowledge_base) or staged_kb:
                    knowledge_base_nonempty = True
                print(f"knowledge_base_nonempty: {knowledge_base_nonempty}")
                # Add knowledge_base_retriever if needed
                if knowledge_base_nonempty and 'knowledge_base_retriever' not in selected_tools:
                    selected_tools.append('knowledge_base_retriever')
                    print("Added 'knowledge_base_retriever' to selected_tools")
                agent.name = name
                agent.role = role
                agent.gender = gender
                agent.personality_traits = traits
                agent.api_key = api_key
                agent.base_prompt = prompt
                agent.voice = voice
                agent.tools = selected_tools
                # If there are staged files, append to knowledge_base
                if staged_kb:
                    if hasattr(agent, 'knowledge_base') and agent.knowledge_base:
                        agent.knowledge_base.extend(staged_kb)
                        print(f"Appended to existing knowledge_base. New length: {len(agent.knowledge_base)}")
                    else:
                        agent.knowledge_base = staged_kb
                        print(f"Set new knowledge_base: {agent.knowledge_base}")
                self.data_manager.save_agent(agent)
                print(f"Agent {agent.name} saved.")
                agent_id_for_ingestion = self.current_editing_agent_id
        else:
            # Create new agent
            temp_agent_id = f"NEW_AGENT_{name.replace(' ', '_')}"
            print(f"Creating new agent. Temp id: {temp_agent_id}")
            if temp_agent_id in self.knowledge_files:
                file_data = self.knowledge_files[temp_agent_id]
                file_paths = file_data['file_paths']
                descriptions = file_data['descriptions']
                print(f"Found staged files for new agent: {file_paths}")
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    description = descriptions.get(file_path, f"Document: {file_name}")
                    staged_kb.append({"doc_name": file_name, "description": description})
                print(f"Staged KB for new agent: {staged_kb}")
           
            # Add knowledge_base_retriever if staged_kb is non-empty
            if staged_kb and 'knowledge_base_retriever' not in selected_tools:
                selected_tools.append('knowledge_base_retriever')
                print("Added 'knowledge_base_retriever' to selected_tools for new agent")
            
            new_agent = Agent.create_new(
                name=name,
                role=role,
                base_prompt=prompt,
                personality_traits=traits,
                api_key=api_key,
                tools=selected_tools,
                gender=gender,
                voice=voice
            )
            # For new agent, set knowledge_base if any staged files
            if staged_kb:
                new_agent.knowledge_base = staged_kb
                print(f"Set knowledge_base for new agent: {new_agent.knowledge_base}")
            self.data_manager.save_agent(new_agent)
            print(f"New agent {name} saved.")
            # After saving, get the new agent's ID for ingestion
            agent_id_for_ingestion = new_agent.id if hasattr(new_agent, 'id') else None
            # Move staged files from temp ID to real agent ID for ingestion
            if agent_id_for_ingestion and temp_agent_id in self.knowledge_files:
                self.knowledge_files[agent_id_for_ingestion] = self.knowledge_files.pop(temp_agent_id)
                print(f"Moved staged files from {temp_agent_id} to {agent_id_for_ingestion} for ingestion.")
        # If there are staged files for this agent, call handle_knowledge_ingestion
        if agent_id_for_ingestion and agent_id_for_ingestion in self.knowledge_files:
            print(f"Calling handle_knowledge_ingestion for agent_id: {agent_id_for_ingestion}")
            ingestion_results = self.handle_knowledge_ingestion(agent_id_for_ingestion)
            # Remove failed docs from knowledge_base
            failed_docs = ingestion_results.get("failed", []) if ingestion_results else []
            failed_msgs = []
            for fail in failed_docs:
                doc_name = fail.get("doc_name")
                reason = fail.get("reason", "Unknown error")
                if doc_name:
                    self.data_manager.remove_document_from_knowledge_base(agent_id_for_ingestion, doc_name)
                    print(f"Removed failed doc from knowledge_base: {doc_name}")
                failed_msgs.append(f"{doc_name or 'Unknown'}: {reason}")
            # Show popup if any failed
            if failed_msgs:
                self.show_failed_ingestion_popup(failed_msgs)
        print("================ SAVE_AGENT END ================\n")


    def handle_knowledge_ingestion(self, agent_id: str):
        """Handles the process of storing and ingesting knowledge base files.
        Returns a dict with lists of successful and failed ingestions (with reasons)."""
        if agent_id not in self.knowledge_files:
            return {"success": [], "failed": []}

        print(f"\n🚀 STARTING KNOWLEDGE INGESTION FOR AGENT {agent_id}")
        print("="*60)

        results = {"success": [], "failed": []}
        try:
            file_data = self.knowledge_files[agent_id]
            file_paths = file_data['file_paths']
            descriptions = file_data['descriptions']

            print(f"📁 Processing {len(file_paths)} files...")

            # Process and ingest the files
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                description = descriptions.get(file_path, f"Document: {file_name}")

                print(f"   📄 Processing: {file_name}")
                print(f"   💬 Description: {description}")

                try:
                    # Use the knowledge manager to process and ingest the file
                    success, message = knowledge_manager.ingest_document_for_agent(
                        agent_id=agent_id,
                        file_path=file_path,
                        description=description
                    )

                    if success:
                        print(f"   ✅ Successfully processed: {file_name}")
                        results["success"].append(file_name)
                    else:
                        print(f"   ❌ Failed to process: {file_name}")
                        results["failed"].append({"doc_name": file_name, "reason": message})

                except Exception as e:
                    print(f"   ❌ Error processing {file_name}: {e}")
                    results["failed"].append({"doc_name": file_name, "reason": str(e)})

            # Clean up staged files
            del self.knowledge_files[agent_id]
            print(f"🧹 Cleaned up staged files for agent {agent_id}")

            print("✅ KNOWLEDGE INGESTION COMPLETED!")
            print("="*60)
            return results

        except Exception as e:
            print(f"❌ KNOWLEDGE INGESTION FAILED: {e}")
            print("="*60)
            results["failed"].append({"doc_name": None, "reason": str(e)})
            return results

    

    def show_failed_ingestion_popup(self, failed_msgs):
        """Show a popup window listing failed document ingestions and reasons."""
        popup = tk.Toplevel(self)
        popup.title("Knowledge Ingestion Failures")
        popup.geometry("600x350")
        popup.grab_set()
        label = tk.Label(popup, text="The following documents failed to ingest:", font=("Arial", 12, "bold"))
        label.pack(pady=(10, 5))
        text = tk.Text(popup, wrap="word", height=10, width=70)
        text.pack(padx=10, pady=5, fill="both", expand=True)
        text.insert("1.0", "\n".join(failed_msgs))
        text.config(state="disabled")

        # Add troubleshooting instructions
        instructions = (
            "\n\nPossible reasons for failure:\n"
            "- File path contains special characters (like the en dash – or em dash — or other Unicode characters) that may cause issues on Windows, especially if the file was renamed, moved, or not supported by the filesystem.\n"
            "- File path is too long for Windows (Windows has a 260-character path limit by default).\n"
            "\nTry renaming the file or moving it to a simpler, shorter path (e.g., C:\\Docs) and avoid special characters."
        )
        instr_label = tk.Label(popup, text=instructions, justify="left", wraplength=560, font=("Arial", 10), fg="#a94442")
        instr_label.pack(padx=10, pady=(0, 10), anchor="w")

        close_btn = ttk.Button(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=(0, 10))
        self.current_editing_agent_id = None
        self.refresh_agents_list()
        self.app.conversation_setup_tab.refresh_agent_checkboxes()
        # Clear the agent details form after saving
        self.clear_agent_form()

    
    
    
    def _update_knowledge_base_tool(self, agent):
        """Update the knowledge_base_retriever tool based on agent's knowledge_base content."""
        has_knowledge_base = hasattr(agent, 'knowledge_base') and agent.knowledge_base and len(agent.knowledge_base) > 0
        has_retriever_tool = 'knowledge_base_retriever' in agent.tools
        
        if has_knowledge_base and not has_retriever_tool:
            agent.tools.append('knowledge_base_retriever')
            print(f"🔧 AUTO-ADDED knowledge_base_retriever tool to agent '{agent.name}' (has {len(agent.knowledge_base)} documents)")
        elif not has_knowledge_base and has_retriever_tool:
            agent.tools.remove('knowledge_base_retriever')
            print(f"🔧 AUTO-REMOVED knowledge_base_retriever tool from agent '{agent.name}' (no documents)")
        elif has_knowledge_base and has_retriever_tool:
            print(f"✅ Agent '{agent.name}' already has knowledge_base_retriever tool ({len(agent.knowledge_base)} documents)")
        else:
            print(f"ℹ️  Agent '{agent.name}' has no knowledge base documents, no retriever tool needed")




    def refresh_agents_list(self):
        """Refresh the agents list in the UI."""
        self.app.agents_listbox.delete(0, tk.END)
        agents = self.data_manager.load_agents()
        for agent in agents:
            display_text = f"{agent.name} ({agent.role})"
            self.app.agents_listbox.insert(tk.END, display_text)

    def load_tool_checkboxes(self):
        """Load available tools and create checkboxes for them."""
        # Clear existing checkboxes
        for widget in self.tools_checkboxes_frame.winfo_children():
            widget.destroy()
        self.tool_vars.clear()
        
        # Load tools from the tools.json file
        try:
            import json
            tools_file = os.path.join(os.path.dirname(__file__), '..', 'tools.json')
            if os.path.exists(tools_file):
                with open(tools_file, 'r', encoding='utf-8') as f:
                    tools_data = json.load(f)
                
                # Handle both old format (flat dict) and new format (nested with "tools" key)
                tools_list = []
                if isinstance(tools_data, dict):
                    if "tools" in tools_data and isinstance(tools_data["tools"], list):
                        # New format with "tools" key containing a list
                        tools_list = tools_data["tools"]
                    else:
                        # Old format - flat dictionary
                        for tool_name, tool_info in tools_data.items():
                            if isinstance(tool_info, dict):
                                tools_list.append({
                                    "name": tool_name,
                                    "description": tool_info.get("description", "No description available")
                                })
                
                row = 0
                for tool_info in tools_list:
                    if isinstance(tool_info, dict):
                        tool_name = tool_info.get("name", "Unknown tool")
                        # Skip knowledge_base_retriever as it's auto-managed
                        if tool_name == 'knowledge_base_retriever':
                            continue
                        
                        var = tk.BooleanVar()
                        self.tool_vars[tool_name] = var
                        
                        checkbox = ttk.Checkbutton(
                            self.tools_checkboxes_frame,
                            text=tool_name,
                            variable=var
                        )
                        checkbox.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                        
                        # Bind events for tooltip functionality
                        description = tool_info.get('description', 'No description available')
                        checkbox.bind("<Enter>", lambda e, desc=description: self.show_tool_tooltip(e.widget, desc))
                        checkbox.bind("<Leave>", lambda e: self.hide_tool_tooltip())
                        
                        row += 1
                    
        except Exception as e:
            print(f"Error loading tools: {e}")
            # Fallback: create checkboxes for known tools
            known_tools = [
                "internet_search_tool",
                "search_internet_strucutred_output", 
                "search_images_from_internet",
                "search_news_from_internet",
                "search_places_from_internet"
            ]
            
            for i, tool_name in enumerate(known_tools):
                var = tk.BooleanVar()
                self.tool_vars[tool_name] = var
                
                checkbox = ttk.Checkbutton(
                    self.tools_checkboxes_frame,
                    text=tool_name,
                    variable=var
                )
                checkbox.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    def show_tool_tooltip(self, widget, text):
        """Show tooltip for tool description."""
        if self.tooltip:
            self.tooltip.destroy()
        
        x = widget.winfo_rootx() + 25
        y = widget.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            self.tooltip,
            text=text,
            justify='left',
            background="#ffffe0",
            relief='solid',
            borderwidth=1,
            wraplength=250,
            font=("Arial", 9)
        )
        label.pack(ipadx=1)

    def hide_tool_tooltip(self):
        """Hide the tool tooltip."""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def update_voice_options(self):
        """Update the voice dropdown based on selected gender."""
        import json
        voices_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kokoro_voices.json")
        gender = self.app.agent_gender_var.get().lower()
        if gender not in ("male", "female"):
            self.app.agent_voice_combo['values'] = ()
            self.app.agent_voice_combo.set('')
            self.app.play_voice_btn['state'] = 'disabled'
            return
        with open(voices_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
        voice_list = voices.get(gender, [])
        self.app.agent_voice_combo['values'] = tuple(voice_list)
        self.app.agent_voice_combo.set('')
        self.app.play_voice_btn['state'] = 'disabled'

    def update_play_button_state(self):
        """Enable play button if a voice is selected."""
        if self.app.agent_voice_var.get():
            self.app.play_voice_btn['state'] = 'normal'
        else:
            self.app.play_voice_btn['state'] = 'disabled'

    def play_selected_voice_sample(self):
        """Play the selected voice sample from the samples directory."""
        import threading
        import platform
        import subprocess
        voice = self.app.agent_voice_var.get()
        gender = self.app.agent_gender_var.get().lower()
        if not voice or gender not in ("male", "female"):
            return
        # Use the correct directory for voice samples
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kokor_voice_samples", "kokor_voice_sample_audios")
        sample_file = os.path.join(sample_dir, f"{voice}_{gender}.wav")
        if not os.path.exists(sample_file):
            messagebox.showerror("Sample Not Found", f"Sample file not found: {sample_file}")
            return
        def play():
            if platform.system() == "Windows":
                import winsound
                winsound.PlaySound(sample_file, winsound.SND_FILENAME)
            else:
                subprocess.run(["aplay", sample_file])
        threading.Thread(target=play, daemon=True).start()
