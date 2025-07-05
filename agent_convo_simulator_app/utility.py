"""
Utility functions for the Agent Conversation Simulator
"""

import re
import json
from typing import Dict, Any, Optional


def extract_json_from_markdown(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from a markdown response string that came from an LLM.
    
    This function handles various formats that LLMs might use to return JSON:
    - JSON wrapped in ```json code blocks
    - JSON wrapped in ``` code blocks without language specification
    - Raw JSON without markdown formatting
    - JSON with additional text before/after
    
    Args:
        response (str): The markdown response string from LLM
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON as dictionary, or None if no valid JSON found
    """
    if not response or not isinstance(response, str):
        return None
    
    # Clean the response string
    response = response.strip()
    
    # Pattern 1: JSON wrapped in ```json code block
    json_pattern1 = r'```json\s*\n?(.*?)\n?\s*```'
    match = re.search(json_pattern1, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: JSON wrapped in ``` code block (without language specification)
    json_pattern2 = r'```\s*\n?(.*?)\n?\s*```'
    matches = re.findall(json_pattern2, response, re.DOTALL)
    
    for match in matches:
        json_str = match.strip()
        # Check if it looks like JSON (starts with { or [)
        if json_str.startswith(('{', '[')):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Pattern 3: Find JSON object starting with { and ending with }
    json_pattern3 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern3, response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Pattern 4: Find JSON array starting with [ and ending with ]
    json_pattern4 = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    matches = re.findall(json_pattern4, response, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            # Convert array to dict if it contains objects
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return {"data": parsed}
            elif isinstance(parsed, list):
                return {"items": parsed}
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Pattern 5: Try to parse the entire response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Pattern 6: Look for JSON-like content between lines
    lines = response.split('\n')
    json_lines = []
    in_json = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('{') or line.startswith('['):
            in_json = True
            json_lines = [line]
        elif in_json:
            json_lines.append(line)
            if line.endswith('}') or line.endswith(']'):
                json_str = '\n'.join(json_lines)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    in_json = False
                    json_lines = []
    
    return None


def extract_json_with_fallback(response: str, fallback_key: str = "content") -> Dict[str, Any]:
    """
    Extract JSON from markdown response with a fallback to wrap the response in a dict.
    
    Args:
        response (str): The markdown response string from LLM
        fallback_key (str): Key to use when wrapping non-JSON response
        
    Returns:
        Dict[str, Any]: Parsed JSON or fallback dict containing the response
    """
    extracted = extract_json_from_markdown(response)
    
    if extracted is not None:
        return extracted
    
    # Fallback: return the response wrapped in a dictionary
    return {fallback_key: response.strip()}


def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by removing common formatting issues.
    
    Args:
        json_str (str): Raw JSON string that might have formatting issues
        
    Returns:
        str: Cleaned JSON string
    """
    if not json_str:
        return json_str
    
    # Remove leading/trailing whitespace
    json_str = json_str.strip()
    
    # Remove markdown code block markers
    json_str = re.sub(r'^```json\s*\n?', '', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'^```\s*\n?', '', json_str)
    json_str = re.sub(r'\n?\s*```$', '', json_str)
    
    # Remove common prefixes that LLMs might add
    prefixes_to_remove = [
        "Here's the JSON:",
        "Here is the JSON:",
        "The JSON response is:",
        "JSON:",
        "Response:",
    ]
    
    for prefix in prefixes_to_remove:
        if json_str.startswith(prefix):
            json_str = json_str[len(prefix):].strip()
    
    return json_str


def validate_json_structure(data: Dict[str, Any], required_keys: list = None) -> bool:
    """
    Validate that a dictionary has the required structure.
    
    Args:
        data (Dict[str, Any]): Dictionary to validate
        required_keys (list): List of required keys
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in data:
                return False
    
    return True


# Example usage and test function
def test_json_extraction():
    """Test function to demonstrate the JSON extraction functionality."""
    
    test_cases = [
        # Case 1: JSON in code block
        '''```json
        {
            "name": "John",
            "age": 30,
            "city": "New York"
        }
        ```''',
        
        # Case 2: JSON with additional text
        '''Here is the response you requested:
        
        ```json
        {
            "status": "success",
            "data": [
                {"id": 1, "value": "test"}
            ]
        }
        ```
        
        I hope this helps!''',
        
        # Case 3: Raw JSON without markdown
        '''{"message": "Hello", "type": "greeting"}''',
        
        # Case 4: JSON in plain code block
        '''```
        {
            "result": "processed",
            "count": 42
        }
        ```''',
        
        # Case 5: Invalid/no JSON
        '''This is just plain text without any JSON content.''',
    ]
    
    print("Testing JSON extraction:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case[:100]}...")
        
        result = extract_json_from_markdown(test_case)
        fallback_result = extract_json_with_fallback(test_case)
        
        print(f"Extracted JSON: {result}")
        print(f"With fallback: {fallback_result}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_json_extraction()
