import os
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load environment variables
env_path = Path('.env')
load_dotenv(env_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = ChatOpenAI(
    openai_api_key=openai_api_key,
    model='gpt-3.5-turbo'
)

def get_completion(messages, max_tokens=500, temperature=0, stop=None, seed=123, logprobs=None, top_logprobs=None, n=None):
    """
    Get a completion from the OpenAI API.
    
    Args:
        messages: The messages to send to the API.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        stop: Sequence where the API will stop generating further tokens.
        seed (int): Seed for random number generator.
        logprobs: Include the log probabilities on the most likely tokens.
        top_logprobs: Return the top log probabilities for each token.
        n (int): Number of completions to generate.
    
    Returns:
        dict: The API response.
    """
    try:
        params = {
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stop': stop,
            'seed': seed,
            'logprobs': logprobs,
            'top_logprobs': top_logprobs
        }
        if n:
            params['n'] = n

        completion = client(**params)
        logger.info("Chat completion successful.")
        return completion
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise

def file_reader(path):
    """
    Read the contents of a file.
    
    Args:
        path (str): The path to the file.
    
    Returns:
        str: The content of the file.
    """
    try:
        if path.endswith('.txt'):
            with open(path, 'r') as f:
                content = f.read()
            logger.info(f"File reading successful for {path}")
        elif path.endswith('.pdf'):
            content = ""
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text()
            logger.info(f"PDF reading successful for {path}")
        else:
            raise ValueError("Unsupported file format")
        return content
    except Exception as e:
        logger.error(f"Error reading the file: {e}")
        raise

def split_text_into_chunks(text, chunk_size=200):
    """
    Split text into chunks of a specified size.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.
    
    Returns:
        list: A list of text chunks.
    """
    try:
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        logger.info(f"Successfully chunked text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking the text: {e}")
        raise
