#from tqdm.auto import tqdm
from data_generation import split_text_into_chunks, file_reader, get_completion
from vectorize import vectorize, retrieve
import pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)


file_path = 'prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf'
# text = file_reader(pdf_path)
# chunks = split_text_into_chunks(text, chunk_size=120)
batch_size=64

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openapi_key = os.getenv('OPENAI_API_KEY')

embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key= openapi_key)

pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-enginerring'

# pc.create_index(
#     name = index_name,
#     dimension=1536,  # Ensure this matches the dimension of your embeddings
#     metric='cosine',
#     spec = ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# while not pc.describe_index(index_name).status['ready']:
#     time.sleep(1)

index = pc.Index(index_name)

vectorize(index, file_path, embed_model, batch_size = 64)

text_field = 'text'
vectorestore = Pinecone(index, embed_model.embed_query, text_field)

query = 'Who are the tutors?'

results = vectorestore.similarity_search(query, k=2)

prompt = retrieve(results, query)

response = get_completion(messages=[HumanMessage(content= prompt)],logprobs=True,top_logprobs=1)

print(response.content)






