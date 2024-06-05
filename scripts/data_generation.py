import os
import json
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


openai_api_key = 'sk-proj-mzvpwbndtOtK0W1r6dElT3BlbkFJC2iTbBxJrYPBUhhImOvo'#os.getenv('OPENAI_API_KEY')

client = ChatOpenAI(
    openai_api_key = openai_api_key,
    model = 'gpt-3.5-turbo'
)


def get_completion(
                   # model,
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
            #'model': model,
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
        with open(filename, 'r') as f:
            system_messages = f.read()

        logger.info('File reading successful for {filename}')

        return system_messages

    except Exception as e:
        logger.error(f"Error reading the file: {e}")









