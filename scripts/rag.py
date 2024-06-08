import os
import logging
from scripts.data_generation import FileHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
import json

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openapi_key = os.getenv('OPENAI_API_KEY')

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

def vectorize(index, file_path, embed_model, batch_size=64):
    """
    Vectorize the contents of a file and upsert the vectors into the given index.
    
    Args:
        index: The Pinecone index to upsert vectors into.
        file_path (str): Path to the file to be vectorized.
        embed_model: The embedding model to use for generating document embeddings.
        batch_size (int): The number of chunks to process in each batch.
    """
    try:
        # Split text into chunks
        chunks = FileHandler.split_text_into_chunks(file_path, chunk_size=120)

        for i in range(0, len(chunks), batch_size):
            i_end = min(len(chunks), i + batch_size)
            batch = chunks[i:i_end]
            ids = [f"{os.path.basename(file_path)}-{j}" for j in range(i, i_end)]
            embeds = embed_model.embed_documents(batch)
            metadata = [{'text': chunk, 'source': file_path} for chunk in chunks]
            index.upsert(vectors=zip(ids, embeds, metadata))

        logger.info(f"Successfully vectorized the data: {index.describe_index_stats()}")
    except Exception as e:
        logger.error(f"Error vectorizing the data: {e}")

def retrieve(results, prompt_path, query=None, num_prompts=None):
    """
    Retrieve the best prompts based on the context of the results.
    
    Args:
        results: The search results from the vector store.
        prompt_path (str): Path to the prompt file.
        query (str): The query to generate prompts for.
        num_prompts (int): The number of prompts to generate.
    
    Returns:
        str: The augmented prompt.
    """
    try:
        context = "\n".join([x.page_content for x in results])
        prompt_text = FileHandler.read_file(prompt_path)

        if num_prompts and query:
            augmented_prompt = prompt_text.replace("{context}", context).replace("{num_prompt_output}", str(num_prompts)).replace("{query}", query) 
        else:
            augmented_prompt = prompt_text.replace("{context}", context).replace("{query}", query) 
        
        return augmented_prompt
    except Exception as e:
        logger.error(f"Error retrieving prompt: {e}")
        raise

def update_json_file(previous_file):
    """
    Update the JSON file with the previous processed file.
    
    Args:
        previous_file (str): Previous processed file.
    """
    try:
        with open('file.json', 'w') as f:
            json.dump({"previous_file": previous_file}, f, indent=4)
    except Exception as e:
        logger.error(f"Error updating JSON file: {e}")
        raise

def insert_vector(text):
    """
    Insert vectorized text into the Pinecone index.
    
    Args:
        text (str): Text to be vectorized and inserted into the index.
    """
    try:
        with open('file.json', 'r') as f:
            data = json.load(f)

        if data['previous_file'] == "" or data['previous_file'] != text[:20]:
            update_json_file(text[:20])
        
            index_name = 'prompt-engineering'
            batch_size = 64

            index = pc.Index(index_name)
            vectorize(index, text, embed_model, batch_size)
    except Exception as e:
        logger.error(f"Error inserting vector: {e}")
        raise


