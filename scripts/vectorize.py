#from tqdm.auto import tqdm
from data_generation import file_reader, split_text_into_chunks
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def vectorize(index, file_path, embed_model, batch_size = 64):

   # pdf_path = '../prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf'
    try:
        text = file_reader(file_path)
        chunks = split_text_into_chunks(text, chunk_size=120)

        for i in range(0, len(chunks), batch_size):
            i_end = min(len(chunks), i+batch_size)
            print(i_end)

            batch = chunks[i:i_end]

            ids = [f"{os.path.basename(file_path)}-{j}" for j in range(i, i_end)]

            embeds = embed_model.embed_documents(batch)
            print(embeds)
            metadata = [{'text': chunk, 'source': file_path} for chunk in chunks]


            index.upsert(vectors=zip(ids, embeds, metadata))

        logger.info(f"Sucessfully vectorizing the data: {index.describe_index_stats()}")

    except Exception as e:
        logger.error(f'Error vectorizing the data {e}')   
    

def retrieve(results, query):
    context = "\n".join([x.page_content for x in results])

    augmented_prompt = f"""Using the contexts below, answer the query
    Contexts:
    {context}
    Query:
    {query}
    """

    return augmented_prompt