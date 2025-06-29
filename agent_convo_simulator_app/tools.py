from langchain_community.utilities import GoogleSerperAPIWrapper
import pprint
import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_core.tools import Tool, tool
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)

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