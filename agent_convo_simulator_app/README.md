# Multi-Agent Conversation Simulator

A desktop application for simulating group conversations between AI agents using LangGraph and Google's Gemini AI model.

## Features

- **Agent Management**: Create, edit, and manage AI agents with custom names, roles, and personalities
- **Conversation Setup**: Define environments and scenes for realistic group conversations
- **Real-time Simulation**: Watch agents interact in real-time with natural conversation flow
- **Scene Control**: Change environments and scenes mid-conversation to explore different dynamics
- **Conversation History**: All conversations are automatically saved to JSON files
- **Memory Management**: Intelligent conversation summarization to maintain context while managing memory usage
- **Pause/Resume**: Control conversation flow with pause and resume functionality
- **Configurable Timing**: Adjust delays between agent responses for different conversation paces

## Requirements

- Python 3.8 or higher
- Google AI API key (get one from [Google AI Studio](https://console.cloud.google.com/))
- Required Python packages (see requirements.txt)

## Installation

1. **Clone or download** this application to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Google AI API key**:
   - Visit [Google AI Studio](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Gemini API
   - Create an API key
   - Keep this key secure - you'll enter it in the application

## Configuration

### Timing Settings

You can customize conversation timing by editing the `config.py` file or using the configuration tool:

```bash
python config_tool.py
```

**Available timing presets:**
- **Fast** (0.5-1.5 seconds): Quick conversations for rapid exchanges
- **Normal** (2-4 seconds): Default balanced timing for natural flow
- **Slow** (5-8 seconds): Thoughtful conversations with longer pauses
- **Very Slow** (10-15 seconds): Deliberate pacing for contemplative discussions

**Configuration options:**
- `start_delay`: Delay before starting conversation (default: 1.0s)
- `agent_turn_delay_min/max`: Random delay range between agent turns (default: 2.0-4.0s)
- `resume_delay`: Delay when resuming paused conversations (default: 1.0s)
- `error_retry_delay`: Delay when retrying after errors (default: 3.0s)
- `max_messages_before_summary`: Message count trigger for summarization (default: 20)
- `messages_to_keep_after_summary`: Messages to retain after summarization (default: 10)

### Manual Configuration

Edit `config.py` directly to customize:
- Gemini model settings
- Temperature values for responses and summaries
- API retry settings
- Message handling parameters

**Note:** Restart the application after changing configuration settings.

## Usage

### Starting the Application

Run the main application:
```bash
python main.py
```

### Creating Agents

1. Go to the **Agents** tab
2. Click "New Agent" to create a new agent
3. Fill in the agent details:
   - **Name**: A unique name for your agent (e.g., "Alice", "Dr. Smith")
   - **Role**: The agent's role or profession (e.g., "Teacher", "Scientist", "Artist")
   - **Personality Traits**: Comma-separated traits (e.g., "curious, friendly, analytical")
   - **Base Prompt**: Detailed description of the agent's personality and behavior
4. Click "Save Agent"

**Example Agent**:
- Name: Emma
- Role: Creative Writer
- Personality Traits: imaginative, enthusiastic, empathetic
- Base Prompt: "You are Emma, a creative writer who loves storytelling and exploring human emotions. You're naturally curious about people's experiences and often draw inspiration from everyday conversations. You tend to ask thought-provoking questions and share creative perspectives on topics."

### Setting Up Conversations

1. Go to the **Conversation Setup** tab
2. Fill in the conversation details:
   - **Title**: A name for your conversation session
   - **Environment**: Where the conversation takes place (e.g., "Coffee Shop", "University Library")
   - **Scene Description**: Detailed description of the setting and atmosphere
3. Select 2-4 agents to participate in the conversation
4. Enter your Google AI API key
5. Click "Start Conversation"

**Example Setup**:
- Title: Book Club Discussion
- Environment: Local Bookstore Café
- Scene Description: "A cozy bookstore café on a Saturday afternoon. Soft jazz music plays in the background, and the smell of fresh coffee fills the air. The participants are sitting around a small wooden table, discussing their latest book club selection."

### Running the Simulation

1. Once started, the simulation will automatically switch to the **Simulation** tab
2. Watch as agents interact naturally based on their personalities and the scene
3. You can:
   - **Send messages**: Type in the input box and press Enter to join the conversation
   - **Pause/Resume**: Control the conversation flow
   - **Change Scene**: Update the environment during the conversation
   - **Summarize**: Generate a summary of the conversation so far
   - **Stop**: End the current conversation

### Changing Scenes

During a conversation, you can change the environment:
1. In the Scene Control panel, enter a new environment and scene description
2. Click "Change Scene"
3. The agents will adapt to the new setting while maintaining conversation continuity

## File Structure

```
agent_convo_simulator_app/
├── main.py                 # Main GUI application
├── data_manager.py         # JSON file management for agents and conversations
├── conversation_engine.py  # LangGraph conversation simulation engine
├── agents.json            # Stored agent configurations
├── conversations.json     # Conversation history and metadata
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data Storage

- **agents.json**: Stores all created agents with their configurations
- **conversations.json**: Stores conversation metadata and message history
- All data is automatically saved and persists between application sessions

## Tips for Better Conversations

1. **Create Diverse Agents**: Mix different personality types and roles for more interesting dynamics
2. **Detailed Prompts**: The more detailed your agent prompts, the more realistic their behavior
3. **Rich Environments**: Detailed scene descriptions help agents stay in character
4. **Natural Interruptions**: Join the conversation as a user to guide or redirect discussions
5. **Scene Changes**: Experiment with changing environments to see how agents adapt

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Google AI API key is valid and has access to the Gemini API
2. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **Slow Responses**: Gemini API responses may vary in speed; be patient during initial model loading
4. **Memory Issues**: The application automatically manages conversation memory, but very long conversations may need manual summarization

### Error Messages

- **"Failed to start conversation"**: Check your API key and internet connection
- **"Invalid Selection"**: Make sure you've selected 2-4 agents for the conversation
- **"Missing Information"**: Fill in all required fields before starting

## Limitations

- Requires active internet connection for AI model access
- Gemini API usage is subject to Google's rate limits and pricing
- Very long conversations may need periodic summarization
- Real-time responses depend on API response times

## Future Enhancements

- Support for additional AI models (OpenAI, Anthropic, etc.)
- Conversation templates and presets
- Export conversations to various formats
- Advanced agent personality customization
- Integration with voice synthesis for audio conversations

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure your Google AI API key is valid and active
4. Check that you have a stable internet connection

## License

This project is provided as-is for educational and experimental purposes.
