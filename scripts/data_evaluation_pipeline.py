import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from scripts.data_generation import AIAssistant, FileHandler
from scripts.rag import retrieve

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openapi_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openapi_key)

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'
index = pc.Index(index_name)

def get_model_confidence(message):
    """
    Get the model's confidence for a given message.

    Args:
        message (str): The message to evaluate.

    Returns:
        str: 'true' if the model has high confidence, otherwise 'false'.

    Raises:
        Exception: If there is an error in the chat completion process.
    """
    try:
        response = assistant.get_chat_completion(message, max_tokens=1, logprobs=True, top_logprobs=1)
        system_msg = str(response.content)

        for i, logprob in enumerate(response.response_metadata['logprobs']['content'][0]['top_logprobs'], start=1):
            output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob["logprob"]}, \naccuracy: {np.round(np.exp(logprob["logprob"])*100,2)}%\n'
            if system_msg == 'true' and np.round(np.exp(logprob["logprob"])*100,2) >= 95.00:
                classification = 'true'
            elif system_msg == 'false' and np.round(np.exp(logprob["logprob"])*100,2) >= 95.00:
                classification = 'false'
            else:
                classification = 'false'
        logger.info(f"Model confidence for message '{message}' is '{classification}'.")
        return classification
    except Exception as e:
        logger.error(f"Error in model confidence evaluation: {e}")
        raise

def evaluation():
    """
    Evaluate and filter data based on model confidence.

    Reads the test dataset, evaluates each entry using the model, and writes back 
    the filtered data to the dataset.

    Raises:
        Exception: If there is an error in the evaluation process.
    """
    try:
        prompt_path = 'prompts/validation.txt'

        with open('test_dataset/test-data.json', 'r') as f:
            data = json.load(f)

        filtered_data = []
        for item in data:
            query = item['user']

            text_field = 'text'
            vectorstore = Pinecone(index, embed_model.embed_query, text_field)
            results = vectorstore.similarity_search(query, k=2)

            prompt = retrieve(results, prompt_path, query)

            classification = get_model_confidence(prompt)
            
            if classification == 'true':
                filtered_data.append(item)
        
        with open('test_dataset/test-data.json', 'w') as file:
            json.dump(filtered_data, file, indent=4)
        
        logger.info("Data evaluation and filtering successful.")
    except Exception as e:
        logger.error(f"Error in data evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluation()
