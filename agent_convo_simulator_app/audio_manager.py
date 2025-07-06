"""
Audio Manager for handling text-to-speech via Kokoro API
"""

import asyncio
import aiohttp
import base64
import io
import threading
import queue
import time
from typing import Dict, Callable, Optional
import pygame


class AudioManager:
    """Manages audio generation and playback for conversation agents."""
    
    def __init__(self, kokoro_api_url: str = "http://localhost:8001"):
        """Initialize the audio manager."""
        self.kokoro_api_url = kokoro_api_url.rstrip('/')
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.running = False
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
            self.pygame_available = True
        except pygame.error as e:
            print(f"Warning: Could not initialize pygame mixer: {e}")
            self.pygame_available = False
        
        # Callbacks for audio events
        self.audio_ready_callback = None
        self.audio_finished_callback = None
        
        # Track current playing audio
        self.current_audio_info = None
        
    def start(self):
        """Start the audio processing thread."""
        if not self.running:
            self.running = True
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
    
    def stop(self):
        """Stop the audio processing thread."""
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
    
    def set_audio_ready_callback(self, callback: Callable[[str, str, str], None]):
        """
        Set callback for when audio is ready.
        Callback signature: (conversation_id, agent_id, message_id)
        """
        self.audio_ready_callback = callback
    
    def set_audio_finished_callback(self, callback: Callable[[str, str, str], None]):
        """
        Set callback for when audio finishes playing.
        Callback signature: (conversation_id, agent_id, message_id)
        """
        self.audio_finished_callback = callback
    
    async def _generate_audio_async(self, text: str, voice: str) -> Optional[bytes]:
        """Generate audio using Kokoro API asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": text,
                    "voice": voice,
                    "return_format": "base64"
                }
                
                async with session.post(
                    f"{self.kokoro_api_url}/synthesize",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        audio_base64 = data.get("audio_base64")
                        if audio_base64:
                            return base64.b64decode(audio_base64)
                    else:
                        print(f"Kokoro API error: {response.status}")
                        return None
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    def _generate_audio_sync(self, text: str, voice: str) -> Optional[bytes]:
        """Generate audio synchronously (wrapper around async method)."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._generate_audio_async(text, voice))
        except Exception as e:
            print(f"Error in sync audio generation: {e}")
            return None
        finally:
            loop.close()
    
    def request_audio(self, conversation_id: str, agent_id: str, message_id: str, text: str, voice: str):
        """Request audio generation for a message."""
        if not self.running:
            self.start()
        
        request = {
            'conversation_id': conversation_id,
            'agent_id': agent_id,
            'message_id': message_id,
            'text': text,
            'voice': voice,
            'timestamp': time.time()
        }
        
        self.audio_queue.put(request)
        print(f"DEBUG: Queued audio request for agent {agent_id}, voice {voice}")
    
    def _audio_worker(self):
        """Worker thread for processing audio requests."""
        while self.running:
            try:
                # Get audio request from queue
                request = self.audio_queue.get(timeout=1.0)
                
                # Generate audio
                audio_data = self._generate_audio_sync(request['text'], request['voice'])
                
                if audio_data:
                    # Notify that audio is ready
                    if self.audio_ready_callback:
                        self.audio_ready_callback(
                            request['conversation_id'],
                            request['agent_id'],
                            request['message_id']
                        )
                    
                    # Play audio if pygame is available
                    if self.pygame_available:
                        self._play_audio(audio_data, request)
                    else:
                        # If no audio playback, still call the finished callback
                        if self.audio_finished_callback:
                            self.audio_finished_callback(
                                request['conversation_id'],
                                request['agent_id'],
                                request['message_id']
                            )
                else:
                    print(f"Failed to generate audio for agent {request['agent_id']}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio worker: {e}")
    
    def _play_audio(self, audio_data: bytes, request: Dict):
        """Play audio data using pygame."""
        try:
            self.current_audio_info = request
            
            # Load audio from bytes
            audio_file = io.BytesIO(audio_data)
            sound = pygame.mixer.Sound(audio_file)
            
            # Play the sound
            channel = sound.play()
            
            # Wait for playback to finish
            while channel.get_busy():
                time.sleep(0.1)
            
            # Notify that audio finished playing
            if self.audio_finished_callback:
                self.audio_finished_callback(
                    request['conversation_id'],
                    request['agent_id'],
                    request['message_id']
                )
            
            self.current_audio_info = None
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.current_audio_info = None
    
    def is_audio_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.current_audio_info is not None
    
    def get_current_playing_agent(self) -> Optional[str]:
        """Get the agent ID of currently playing audio."""
        if self.current_audio_info:
            return self.current_audio_info['agent_id']
        return None
    
    def clear_pending_audio(self, conversation_id: str) -> list:
        """
        Clear all pending audio requests for a conversation.
        Returns a list of cleared requests with their message IDs.
        """
        cleared_requests = []
        temp_queue = queue.Queue()
        
        # Extract all items from the queue
        while not self.audio_queue.empty():
            try:
                request = self.audio_queue.get_nowait()
                if request['conversation_id'] == conversation_id:
                    # This request should be cleared
                    cleared_requests.append({
                        'agent_id': request['agent_id'],
                        'message_id': request['message_id']
                    })
                else:
                    # Keep requests from other conversations
                    temp_queue.put(request)
            except queue.Empty:
                break
        
        # Put back the requests we want to keep
        while not temp_queue.empty():
            try:
                request = temp_queue.get_nowait()
                self.audio_queue.put(request)
            except queue.Empty:
                break
        
        print(f"DEBUG: Cleared {len(cleared_requests)} pending audio requests for conversation {conversation_id}")
        return cleared_requests
    
    def get_current_playing_info(self) -> Optional[Dict]:
        """Get information about currently playing audio."""
        return self.current_audio_info
