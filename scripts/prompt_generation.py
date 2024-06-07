import os
import time
import logging
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import HumanMessage
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from rag import vectorize, retrieve, insert_vector
from data_generation import FileHandler, AIAssistant


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')

openapi_key = os.getenv('OPENAI_API_KEY')

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openapi_key)

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'
index = pc.Index(index_name)

def create_pinecone_index(pc, index_name):
    """
    Create a Pinecone index if it doesn't already exist.
    
    Args:
        pc: The Pinecone client.
        index_name (str): The name of the index to create.
    """
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ensure this matches the dimension of your embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

def generation_prompt(text, query, prompt_path, n=5):

    insert_vector(text)

    # Initialize the vector store
    text_field = 'text'
    vectorstore = Pinecone(index, embed_model.embed_query, text_field)

    # Define the query
    #query = 'Who are the tutors?'

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=2)

    # Retrieve the prompt
    prompt = retrieve(results, query, prompt_path, n)

    # Get completion from OpenAI
    response = assistant.get_chat_completion(messages=[HumanMessage(content=prompt)], logprobs=True, top_logprobs=1)

    return response

if __name__ == "__main__":
    file_path = 'prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf'
    prompt_path = 'prompts/prompt-generation.txt'
    text = FileHandler.read_file(file_path)
    query = 'Who are the tutors?'

    response = generation_prompt(text, prompt_path, query, 1)
    print(response.content)
