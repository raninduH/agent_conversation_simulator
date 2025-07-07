import tkinter as tk
from tkinter import ttk, messagebox

class PastConversationsTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager

        self.create_widgets()

    def create_widgets(self):
        """Create the past conversations tab."""
        past_conv_frame = ttk.Frame(self)
        past_conv_frame.pack(fill="both", expand=True)

        # Configure grid
        past_conv_frame.grid_rowconfigure(1, weight=1)
        past_conv_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(past_conv_frame, text="Past Conversations", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 20), sticky="w", padx=10)

        # Main content frame
        content_frame = ttk.Frame(past_conv_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Listbox for past conversations
        list_frame = ttk.LabelFrame(content_frame, text="Saved Conversations", padding="10")
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        list_container = tk.Frame(list_frame)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        self.past_conv_listbox = tk.Listbox(list_container)
        self.past_conv_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.past_conv_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.past_conv_listbox.configure(yscrollcommand=scrollbar.set)

        # Buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.grid(row=1, column=0, pady=(10, 0), sticky="e")

        ttk.Button(btn_frame, text="Load Conversation", command=self.load_selected_conversation_from_tab).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Delete Conversation", command=self.delete_selected_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_past_conversations).pack(side=tk.LEFT, padx=5)

    def refresh_past_conversations(self):
        """Refresh the list of past conversations."""
        self.past_conv_listbox.delete(0, tk.END)
        conversations = self.data_manager.load_conversations()
        for conv in conversations:
            agents_count = len(conv.agents)
            status = f" ({conv.status})" if hasattr(conv, 'status') and conv.status else ""
            display_text = f"{conv.title} - {conv.environment} ({agents_count} agents) - {conv.created_at[:10]}{status}"
            self.past_conv_listbox.insert(tk.END, display_text)

    def load_selected_conversation_from_tab(self):
        """Load the selected conversation from the past conversations tab."""
        selection = self.past_conv_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation to load.")
            return
        
        conversations = self.data_manager.load_conversations()
        if selection[0] < len(conversations):
            conversation = conversations[selection[0]]
            self.app.load_selected_conversation(conversation)

    def delete_selected_conversation(self):
        """Delete the selected conversation."""
        selection = self.past_conv_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation to delete.")
            return

        conversations = self.data_manager.load_conversations()
        if selection[0] < len(conversations):
            conversation = conversations[selection[0]]
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete conversation '{conversation.title}'?"):
                self.data_manager.delete_conversation(conversation.id)
                self.refresh_past_conversations()
                self.app.update_status(f"Conversation '{conversation.title}' deleted.")
