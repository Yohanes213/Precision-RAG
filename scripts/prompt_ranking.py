from langchain.schema import HumanMessage
from scripts.data_generation import AIAssistant
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import HumanMessage
from pinecone import Pinecone as PineconeClient
from scripts.rag import retrieve
from sklearn.metrics.pairwise import cosine_similarity
import json

pinecone_api_key = os.getenv('PINECONE_API_KEY')

openapi_key = os.getenv('OPENAI_API_KEY')

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openapi_key)

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'
index = pc.Index(index_name)

import json

def load_validation_dataset(file_path):
    with open(file_path, 'r') as f:
        validation_dataset = json.load(f)
    return validation_dataset


def generate_response(prompt, embed_model, index, prompt_path):

    text_field = 'text'
    vectorstore = Pinecone(index, embed_model.embed_query, text_field)

    # Define the query
    #query = 'Who are the tutors?'

    # Perform similarity search
    results = vectorstore.similarity_search(prompt, k=2)

    # Retrieve the prompt
    aug_prompt = retrieve(results,  prompt_path, prompt)

    #prompts = prompts.path(results, )

    # Generate a response using the model
    response = assistant.get_chat_completion(messages=[HumanMessage(content=aug_prompt)], logprobs=True, top_logprobs=1).content

    return response


def evaluate_response(prompt, generated_response, expected_responses):
    # Embed the generated response
    generated_vector = embed_model.embed_query(generated_response)

    # Embed all expected responses and calculate their similarities
    scores = []
    for expected_response in expected_responses:
        expected_vector = embed_model.embed_query(expected_response)
        relevance = cosine_similarity([generated_vector], [expected_vector])[0][0]
        clarity = len(generated_response.split()) / 100
        completeness = len(set(prompt.split()) & set(generated_response.split())) / len(set(prompt.split()))
        score = relevance + clarity + completeness
        scores.append(score)

    # Return the average score
    return sum(scores) / len(scores)


def expected_score(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))


def update_ratings(rating1, rating2, result, k=32):
    expected1 = expected_score(rating1, rating2)
    expected2 = expected_score(rating2, rating1)

    if result == 1:  # prompt1 wins
        rating1 += k * (1 - expected1)
        rating2 += k * (0 - expected2)
    else:  # prompt2 wins
        rating1 += k * (0 - expected1)
        rating2 += k * (1 - expected2)

    return rating1, rating2



def ranking_prompts(prompts, prompt_path):#, validation_dataset):
    validation_dataset = load_validation_dataset('test_dataset/test-data.json')
    responses = {prompt: generate_response(prompt, embed_model, index, prompt_path) for prompt in prompts}
    scores = {}

    for prompt in prompts:
        total_score = 0
        for data in validation_dataset:
            expected_responses = [data['assistant']]
            generated_response = responses[prompt]
            #evaluate_response(prompt, response, document_vector, embed_model)
            score = evaluate_response(prompt, generated_response, expected_responses)#, embed_model)
            total_score += score
        scores[prompt] = total_score / len(validation_dataset)

    ratings = {prompt: 1000 for prompt in prompts}
    comparisons = [(prompts[i], prompts[j]) for i in range(len(prompts)) for j in range(i+1, len(prompts))]

    for prompt1, prompt2 in comparisons:
        rating1 = ratings[prompt1]
        rating2 = ratings[prompt2]
        score1 = scores[prompt1]
        score2 = scores[prompt2]

        result = 1 if score1 > score2 else 0
        new_rating1, new_rating2 = update_ratings(rating1, rating2, result)
        ratings[prompt1] = new_rating1
        ratings[prompt2] = new_rating2

    ranked_prompts = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    return ranked_prompts

if __name__ == "__main__":
  
    prompts = ["1. Who are the team tutors for this week's challenge in Automatic Prompt Engineering?",
    '2. Which individuals are responsible for tutoring participants struggling with prompt engineering concepts?',
    '3. Who are the designated tutors for providing guidance and support in prompt optimization strategies?',
    '4. Who are the mentors available to assist with understanding prompt engineering tools and concepts?',
    '5. Who are the experts assigned to help others in the community with prompt engineering tasks?']
  
    prompt_path = 'prompts/response.txt'

    answer = ranking_prompts(prompts, prompt_path)

    print(answer)
