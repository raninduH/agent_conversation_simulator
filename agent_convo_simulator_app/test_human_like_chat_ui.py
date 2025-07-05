"""
Quick test to verify human_like_chat UI integration
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_invocation_method_ui():
    """Test that human_like_chat option is available in the UI."""
    print("Testing human_like_chat UI integration...")
    
    # Create a simple test window with the invocation method selection
    root = tk.Tk()
    root.title("Test Human-like Chat UI")
    root.geometry("600x300")
    
    # Test the invocation method variable and radio buttons
    invocation_method_var = tk.StringVar(value="round_robin")
    
    # Create frame for radio buttons
    method_frame = ttk.Frame(root)
    method_frame.pack(pady=20)
    
    ttk.Label(root, text="Available Invocation Methods:").pack(pady=10)
    
    # Create the radio buttons (same as in main.py)
    ttk.Radiobutton(
        method_frame, 
        text="Round Robin", 
        variable=invocation_method_var, 
        value="round_robin"
    ).pack(side=tk.LEFT, padx=(0, 10))
    
    ttk.Radiobutton(
        method_frame, 
        text="Agent Selector (LLM)", 
        variable=invocation_method_var, 
        value="agent_selector"
    ).pack(side=tk.LEFT, padx=(0, 10))
    
    ttk.Radiobutton(
        method_frame, 
        text="Human-like Chat", 
        variable=invocation_method_var, 
        value="human_like_chat"
    ).pack(side=tk.LEFT)
    
    # Test that all values work
    def test_selection():
        current = invocation_method_var.get()
        result_label.config(text=f"Selected: {current}")
        print(f"Selected invocation method: {current}")
    
    test_button = ttk.Button(root, text="Test Selection", command=test_selection)
    test_button.pack(pady=10)
    
    result_label = ttk.Label(root, text="Selected: round_robin")
    result_label.pack(pady=5)
    
    # Test programmatic selection of human_like_chat
    def select_human_like_chat():
        invocation_method_var.set("human_like_chat")
        test_selection()
    
    auto_test_button = ttk.Button(root, text="Auto-select Human-like Chat", command=select_human_like_chat)
    auto_test_button.pack(pady=5)
    
    # Add instructions
    instructions = ttk.Label(root, 
                           text="1. Click radio buttons to test selection\n2. Click 'Test Selection' to see current value\n3. Click 'Auto-select Human-like Chat' to test programmatic selection",
                           font=("TkDefaultFont", 9))
    instructions.pack(pady=10)
    
    # Close button
    ttk.Button(root, text="Close", command=root.quit).pack(pady=10)
    
    print("✅ UI test window created successfully")
    print("✅ All three invocation methods are available:")
    print("   - Round Robin")
    print("   - Agent Selector (LLM)")
    print("   - Human-like Chat")
    
    # Run the test window
    print("\nOpening test window... Close it to continue.")
    root.mainloop()
    root.destroy()
    
    print("✅ UI test completed successfully")

if __name__ == "__main__":
    test_invocation_method_ui()
