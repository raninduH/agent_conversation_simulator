from langchain_community.utilities import GoogleSerperAPIWrapper
import pprint
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_core.tools import Tool, tool
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)
from typing import List, Dict, Any
from .knowledge_manager import KnowledgeManager

load_dotenv()


search = GoogleSerperAPIWrapper()

internet_search_tool = Tool(
        name="Intermediate_Answer",
        func=search.run,
        description="useful for when you need to ask with search. You can search the internet for information to answer questions about current events or general knowledge. Input should be a search query.",
    )

@tool("search_internet_strucutred_output")
def search_internet_strucutred_output(query: str) -> dict:
    """Search the internet and return structured output with comprehensive search results.

    Args:
        query: the query to search for information.
    """
    results = search.results(query)
    return results


@tool("search_images_from_internet")
def search_images_from_internet(query: str) -> dict:
    """Search the internet for images on a specific query and return structured results.

    Args:
        query: The query to search for images.
    """
    search = GoogleSerperAPIWrapper(type="images")
    results = search.results(query)
    return results

@tool("search_news_from_internet")
def search_news_from_internet(query: str, past_period:str = "qdr:h") -> dict:
    """Search the internet for news articles on a specific query and return structured results.

    Args:
        query: The query to search for news articles.
        past_period: specify the time period for news articles (e.g., "past week", "past month").
        qdr:h (past hour) qdr:d (past day) qdr:w (past week) qdr:m (past month) qdr:y (past year)
        You can specify intermediate time periods by adding a number: qdr:h12 (past 12 hours) qdr:d3 (past 3 days) qdr:w2 (past 2 weeks) qdr:m6 (past 6 months) qdr:m2 (past 2 years)
    """
    
    search = GoogleSerperAPIWrapper(type="news", tbs=past_period)
    results = search.results(query)
    return results


@tool("search_places_from_internet")
def search_places_from_internet(query: str) -> dict:
    """Search the internet for places on a specific topic and return structured results.

    Args:
        query: The query to search for places (e.g., restaurants, businesses, landmarks in a location).
    """
    search = GoogleSerperAPIWrapper(type="places")
    results = search.results(query)
    return results



# Global variable to store the knowledge manager (lazy loading)
knowledge_manager = None

def get_knowledge_manager():
    """Lazily loads and returns the knowledge manager."""
    global knowledge_manager
    if knowledge_manager is None:
        knowledge_manager = KnowledgeManager()
    return knowledge_manager

@tool("knowledge_base_retriever")
def knowledge_base_retriever(query: str, agent_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the top 3 most relevant results from the agent's knowledge base.

    Args:
        query: The query to search for in the knowledge base.
        agent_id: The ID of the agent performing the search.

    Returns:
        A list of dictionaries, where each dictionary contains the content
        and score of a relevant document.
    """
    print(f"\n🔍 KNOWLEDGE BASE SEARCH INITIATED")
    print(f"   🤖 Agent: {agent_id}")
    print(f"   🔎 Query: '{query}'")
    print(f"   📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not agent_id:
        print(f"   ❌ Error: agent_id must be provided")
        return [{"error": "agent_id must be provided"}]
    
    if not query or not query.strip():
        print(f"   ❌ Error: query cannot be empty")
        return [{"error": "query cannot be empty"}]
    
    try:
        # The agent_id corresponds to the Pinecone index name
        # Convert to proper index format (lowercase, with dashes)
        index_name = f"agent-kb-{agent_id.lower().replace('_', '-')}"
        print(f"   📋 Target index: {index_name}")
        
        # Query the knowledge base
        print(f"   ⏳ Searching knowledge base...")
        search_start = time.time()
        
        km = get_knowledge_manager()
        results = km.query_pinecone(index_name, query, top_k=3)
        
        search_time = time.time() - search_start
        print(f"   ⏱️  Search completed in {search_time:.2f} seconds")
        print(f"   📊 Results found: {len(results)}")
        
        if not results:
            no_results_msg = "No relevant information found in the knowledge base."
            print(f"   ℹ️  {no_results_msg}")
            return [{"message": no_results_msg}]
        
        # Log result summary
        print(f"   ✅ Returning {len(results)} relevant result(s):")
        for i, result in enumerate(results, 1):
            score = result.get('score', 0.0)
            content_preview = result.get('content', '')[:100] + '...' if len(result.get('content', '')) > 100 else result.get('content', '')
            print(f"      {i}. Score: {score:.4f} | Preview: {content_preview}")
            
        return results
        
    except Exception as e:
        error_msg = f"Failed to retrieve information from knowledge base. {str(e)}"
        print(f"   ❌ Search failed: {error_msg}")
        import traceback
        traceback.print_exc()
        return [{"error": error_msg}]


# Lazy loading for browser toolkit to avoid async/threading issues
browser_manipulation_toolkit = None

def get_browser_tools():
    """Get browser manipulation tools with lazy loading to avoid async/threading issues."""
    global browser_manipulation_toolkit
    
    if browser_manipulation_toolkit is None:
        try:
            import asyncio
            # Check if we're in the main thread and can create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async_browser = create_async_playwright_browser()
            toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
            browser_manipulation_toolkit = toolkit.get_tools()
        except Exception as e:
            print(f"Warning: Could not load browser tools: {e}")
            # Return empty list if browser tools can't be loaded
            browser_manipulation_toolkit = []
    
    return browser_manipulation_toolkit