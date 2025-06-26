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

@tool("search_internet_strucutred_output", parse_docstring=True)
def search_internet_strucutred_output(query: str) -> dict:
    """Search the internet and return structured output with comprehensive search results.

    Args:
        query: the query to search for information.
    Returns:
        A dictionary containing the structured search results with the following components:
        - searchParameters: Information about the search query and configuration
        - knowledgeGraph: Entity information including title, type, description, image, and attributes (when available)
        - organic: List of regular search results with title, link, snippet, and position
        - images: List of image results with title, imageUrl, and link
        - peopleAlsoAsk: Related questions with their answers, titles and links
        - relatedSearches: List of related search queries
        - sitelinks: For important websites, additional deep links may appear under organic results
        - infobox: Detailed structured information about entities (when available)
        - answerBox: Direct answers to questions (when available)
        - pagination: Information about result pages
        - credits: Number of API credits used for this search
    """
    results = search.results(query)
    return results


@tool("search_images_from_internet", parse_docstring=True)
def search_images_from_internet(query: str) -> dict:
    """Search the internet for images on a specific query and return structured results.

    Args:
        query: The query to search for images.
    Returns:
        A dictionary containing structured image search results with the following components:
        - searchParameters: Information about the search query and configuration (q, gl, hl, type, num, engine)
        - images: List of image results, each containing:
          - title: The title of the page containing the image
          - imageUrl: Direct URL to the full-sized image
          - imageWidth: Width of the original image in pixels
          - imageHeight: Height of the original image in pixels
          - thumbnailUrl: URL to a smaller thumbnail version of the image
          - thumbnailWidth: Width of the thumbnail in pixels
          - thumbnailHeight: Height of the thumbnail in pixels
          - source: The name of the website or source of the image
          - domain: The domain name of the source website
          - link: URL to the webpage containing the image
          - googleUrl: Google's redirect URL for the image
          - position: Ranking position in the search results
          - creator: Original creator of the image (when available)
          - copyright: Copyright information (when available)
          - credit: Attribution information (when available)
        - credits: Number of API credits used for this search"""
    search = GoogleSerperAPIWrapper(type="images")
    results = search.results(query)
    return results

@tool("search_news_from_internet", parse_docstring=True)
def search_news_from_internet(query: str, past_period:str = "qdr:h") -> dict:
    """Search the internet for news articles on a specific query and return structured results.

    Args:
        query: The query to search for news articles.
        past_period: specify the time period for news articles (e.g., "past week", "past month").
        qdr:h (past hour) qdr:d (past day) qdr:w (past week) qdr:m (past month) qdr:y (past year)
        You can specify intermediate time periods by adding a number: qdr:h12 (past 12 hours) qdr:d3 (past 3 days) qdr:w2 (past 2 weeks) qdr:m6 (past 6 months) qdr:m2 (past 2 years)
    
    Returns:
        A dictionary containing structured news search results with the following components:
        - searchParameters: Information about the search query and configuration
        - news: List of news article results, each containing:
          - title: The title of the news article
          - link: URL to the news article
          - snippet: A brief excerpt from the news article
          - source: The name of the news source
          - date: Publication date (when available)
          - imageUrl: URL to an image associated with the article (when available)
          - thumbnail: Thumbnail image information (when available)
          - position: Ranking position in the search results
        - credits: Number of API credits used for this search"""
    
    search = GoogleSerperAPIWrapper(type="news", tbs=past_period)
    results = search.results(query)
    return results


@tool("search_places_from_internet", parse_docstring=True)
def search_places_from_internet(query: str) -> dict:
    """Search the internet for places on a specific topic and return structured results.

    Args:
        query: The query to search for places (e.g., restaurants, businesses, landmarks in a location).
    Returns:
        A dictionary containing structured place search results with the following components:
        - searchParameters: Information about the search query and configuration (q, gl, hl, type, num, engine)
        - places: List of place results, each containing:
          - position: Ranking position in the search results
          - title: The name of the place
          - address: The full address of the place
          - latitude: Geographic latitude coordinate
          - longitude: Geographic longitude coordinate
          - rating: Average customer rating (typically on a 1-5 scale)
          - ratingCount: Number of customer reviews/ratings
          - priceLevel: Price level indicator (e.g., $, $$, $$$, or price range like $30-50)
          - category: Type of place or business category
          - phoneNumber: Contact telephone number
          - website: URL to the business website (when available)
          - cid: Google's unique identifier for the place
        - credits: Number of API credits used for this search"""
    search = GoogleSerperAPIWrapper(type="places")
    results = search.results(query)
    return results



async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
browser_manipulation_toolkit = toolkit.get_tools()