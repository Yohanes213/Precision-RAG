import os
import logging
from data_generation import file_reader, split_text_into_chunks
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
        #text = file_reader(file_path)
        chunks = split_text_into_chunks(file_path, chunk_size=120)

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

def retrieve(results, query, num_prompts):
    """
    Retrieve the best prompts based on the context of the results.
    
    Args:
        results: The search results from the vector store.
        query (str): The query to generate prompts for.
        num_prompts (int): The number of prompts to generate.
    
    Returns:
        str: The augmented prompt.
    """
    context = "\n".join([x.page_content for x in results])

    prompt_path = 'prompts/prompt-generation.txt'
    
    prompt_text = file_reader(prompt_path)
    augmented_prompt = prompt_text.replace("{context}", context).replace("{num_prompt_output}", str(num_prompts)).replace("{query}", query) 
    return augmented_prompt

