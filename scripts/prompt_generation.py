import os
import time
import logging
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import HumanMessage
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from vectorize import vectorize, retrieve
from data_generation import get_completion, file_reader

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

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'

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

def generation_prompt(file_path, query, n=5):
    # File path to the document
    #file_path = 'prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf'
    batch_size = 64

    # Create Pinecone index
    #create_pinecone_index(pc, index_name)
    index = pc.Index(index_name)

    # Vectorize the document
    vectorize(index, file_path, embed_model, batch_size)

    # Initialize the vector store
    text_field = 'text'
    vectorstore = Pinecone(index, embed_model.embed_query, text_field)

    # Define the query
    #query = 'Who are the tutors?'

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=2)

    # Retrieve the prompt
    prompt = retrieve(results, query, n)

    # Get completion from OpenAI
    response = get_completion(messages=[HumanMessage(content=prompt)], logprobs=True, top_logprobs=1)

    return response

if __name__ == "__main__":
    file_path = 'prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf'
    text = file_reader(file_path)
    query = 'Who are the tutors?'
    response = generation_prompt(text, query, 1)
    print(response.content)
