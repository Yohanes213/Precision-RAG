import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class FileHandler:
    def read_file(path):
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

class AIAssistant:
    def __init__(self, openai_api_key):
        self.client = ChatOpenAI(
            openai_api_key=openai_api_key,
            model='gpt-3.5-turbo'
        )

    def get_chat_completion(self, messages, max_tokens=500, temperature=0, stop=None, seed=123, logprobs=None, top_logprobs=None, n=None):
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
                'input': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stop': stop,
                'seed': seed,
                'logprobs': logprobs,
                'top_logprobs': top_logprobs
            }
            if n:
                params['n'] = n

            completion = self.client.invoke(**params)
            logger.info("Chat completion successful.")
            return completion
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def get_message_classification(self, message):
        try:
            response = self.client.invoke(message, max_tokens=1, logprobs=True, top_logprobs=1)
            system_msg = str(response.content)
            for logprob in response.response_metadata['logprobs']['content'][0]['top_logprobs']:
                output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob["logprob"]}, \naccuracy: {np.round(np.exp(logprob["logprob"])*100,2)}%\n'
                if system_msg == 'true' and np.round(np.exp(logprob["logprob"])*100,2) >= 95.00:
                    classification = 'true'
                elif system_msg == 'false' and np.round(np.exp(logprob["logprob"])*100,2) >= 95.00:
                    classification = 'false'
                else:
                    classification = 'false'
            return classification
        except Exception as e:
            logger.error(f"Error in message classification: {e}")
            raise

