import os
import json
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

env_path = Path('.env')
load_dotenv(env_path)

openai_api_key = os.getenv('OPENAI_API_KEY')

client = ChatOpenAI(
    openai_api_key = openai_api_key,
    model = 'gpt-3.5-turbo'
)


def get_completion(
                    messages, 
                    max_tokens=500, 
                    temperature=0, 
                    stop=None,
                    seed=123,
                    tools=None,
                    logprobs=None,
                    top_logprobs=None):
    
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

        if tools:
            params['tools'] = tools
        completion = client(**params) 

        logger.info(f"Chat completion successfull.")

        return completion
    except Exception as e:
        logger.error(f'Error in chat completion: {e}')
        raise

def file_reader(path):

    try:
        filename = os.path.join(path)
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                content = f.read()
            logger.info(f'File reading successful for {filename}')
        elif filename.endswith('.pdf'):
            content = ""
            with open(filename, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text()
            logger.info(f'PDF reading successful for {filename}')
        else:
            raise ValueError('Unsupported file format')
        
        return content

    except Exception as e:
        logger.error(f"Error reading the file: {e}")
        raise










