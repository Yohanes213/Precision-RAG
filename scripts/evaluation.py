from data_generation import AIAssistant, FileHandler
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from rag import retrieve

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

def get_model_confidence(message):

    #response = client.invoke(message, max_tokens=1, logprobs=True, top_logprobs=1)

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
    return classification


if __name__ == "__main__":
    prompt_path = 'prompts/validation.txt'
    query ="Who is the doctor?"
    n = 5
    text_field = 'text'
    vectorstore = Pinecone(index, embed_model.embed_query, text_field)
    results = vectorstore.similarity_search(query, k=2)
#results, query, prompt_path, num_prompts=None)
    prompt = retrieve(results, query, prompt_path)

    classficaion = get_model_confidence(prompt)

    print(classficaion)


