import json
import requests
import time
import os
from urllib.parse import urlencode

API_KEY = "AIzaSyBwfDBo21ftJpTMouehfBUPgcOzl0QmEoU"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

"""Define the chatbot's role, memory, and utility functions."""

SYSTEM_PROMPT = (
    "You are 'HITS Bot', an official, friendly, and highly informative chatbot "
    "for Karmaveer bhaurao patil college vashi, India. "
    "Your primary goal is to provide accurate and helpful information about the "
    "university, including admissions, courses, campus life, faculty, and recent news. "
    "Keep your answers concise, professional, and encouraging. Always maintain "
    "the persona of a representative of HITS."
)
chat_history = []

def call_gemini_api(prompt, system_instruction, history, tools=None):
    """
    Calls the Gemini API with exponential backoff for text generation.

    Args:
        prompt (str): The user's latest query.
        system_instruction (str): The model's persona definition.
        history (list): List of previous messages for context.
        tools (list, optional): Tools for grounding (e.g., Google Search).

    Returns:
        str: The generated response text, or an error message.
    """
    full_contents = []
    for message in history:
        full_contents.append({
            "role": message['role'],
            "parts": [{"text": message['text']}]
        })
    
    full_contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    
    payload = {
        "contents": full_contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        }
    }

    if tools:
        payload['tools'] = tools

    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        try:
            
            url = f"{API_URL}?key={API_KEY}"
            
            response = requests.post(
                url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() 
            result = response.json()
            
            
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0]['text']

          
            return "Error: Could not extract response text from API or the response was blocked."

        except requests.exceptions.HTTPError as e:
            
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2 
                continue
            return f"HTTP Error: {e} - Response: {response.text}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    return "Error: Max retries reached. The API is unresponsive."

def update_history(role, text):
    """
    Adds a message to the global chat history.
    Limits history size to 10 messages (5 turns).
    """
    chat_history.append({'role': role, 'text': text})
    if len(chat_history) > 10:
        chat_history.pop(0)

"""Main interactive loop for the chatbot."""

def run_chatbot():
    """
    Initializes and runs the interactive HITS Chatbot.
    """
    print("----------------------------------------------------------------------")
    print("Welcome to HITS Bot! I am here to answer your questions about the")
    print("Hindustan Institute of Technology and Science.")
    print("Type 'exit' or 'quit' to end the session.")
    print("----------------------------------------------------------------------")

    
    tools_config = [
        {
            "googleSearch": {}
        }
    ]

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nHITS Bot: Thank you for chatting! Have a great day.")
                break
            
            if not user_input.strip():
                continue
            update_history("user", user_input)
            
            response_text = call_gemini_api(
                prompt=user_input,
                system_instruction=SYSTEM_PROMPT,
                history=chat_history,
                tools=tools_config 
            )
            
            update_history("model", response_text)

            print(f"HITS Bot: {response_text}\n")

        except EOFError:
            print("\nExiting chat.")
            break
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"\nAn unexpected runtime error occurred: {e}")
            break

if __name__ == "__main__":
    run_chatbot()