import os
import logging
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from scripts.data_generation import AIAssistant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from scripts.rag import retrieve
from sklearn.metrics.pairwise import cosine_similarity
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

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openapi_key)

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'
index = pc.Index(index_name)

def load_validation_dataset(file_path):
    """
    Load the validation dataset from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: The loaded validation dataset.
    """
    try:
        with open(file_path, 'r') as f:
            validation_dataset = json.load(f)
        logger.info(f"Loaded validation dataset from {file_path}")
        return validation_dataset
    except Exception as e:
        logger.error(f"Error loading validation dataset: {e}")
        raise

def generate_response(prompt, embed_model, index, prompt_path):
    """
    Generate a response for a given prompt.

    Args:
        prompt (str): The prompt for response generation.
        embed_model: The embedding model.
        index: The Pinecone index.
        prompt_path (str): The path to the prompt file.

    Returns:
        str: The generated response.
    """
    try:
        text_field = 'text'
        vectorstore = Pinecone(index, embed_model.embed_query, text_field)

        results = vectorstore.similarity_search(prompt, k=2)

        aug_prompt = retrieve(results, prompt_path, prompt)

        response = assistant.get_chat_completion(messages=[HumanMessage(content=aug_prompt)], logprobs=True, top_logprobs=1).content

        logger.info(f"Generated response for prompt: {prompt}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

def evaluate_response(prompt, generated_response, expected_responses):
    """
    Evaluate a generated response against expected responses.

    Args:
        prompt (str): The prompt used for response generation.
        generated_response (str): The generated response.
        expected_responses (list): List of expected responses.

    Returns:
        float: The average score of the generated response.
    """
    try:
        generated_vector = embed_model.embed_query(generated_response)
        scores = []
        for expected_response in expected_responses:
            expected_vector = embed_model.embed_query(expected_response)
            relevance = cosine_similarity([generated_vector], [expected_vector])[0][0]
            clarity = len(generated_response.split()) / 100
            completeness = len(set(prompt.split()) & set(generated_response.split())) / len(set(prompt.split()))
            score = relevance + clarity + completeness
            scores.append(score)
        return sum(scores) / len(scores)
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        raise

def expected_score(rating1, rating2):
    """
    Calculate the expected score.

    Args:
        rating1 (float): Rating of prompt 1.
        rating2 (float): Rating of prompt 2.

    Returns:
        float: Expected score.
    """
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_ratings(rating1, rating2, result, k=32):
    """
    Update ratings based on comparison results.

    Args:
        rating1 (float): Rating of prompt 1.
        rating2 (float): Rating of prompt 2.
        result (int): Result of comparison (1 if prompt1 wins, 0 otherwise).
        k (int, optional): A constant value. Defaults to 32.

    Returns:
        tuple: Updated ratings for prompt 1 and prompt 2.
    """
    try:
        expected1 = expected_score(rating1, rating2)
        expected2 = expected_score(rating2, rating1)

        if result == 1:  # prompt1 wins
            rating1 += k * (1 - expected1)
            rating2 += k * (0 - expected2)
        else:  # prompt2 wins
            rating1 += k * (0 - expected1)
            rating2 += k * (1 - expected2)

        return rating1, rating2
    except Exception as e:
        logger.error(f"Error updating ratings: {e}")
        raise

def ranking_prompts(prompts, prompt_path):
    """
    Rank prompts based on the generated responses.

    Args:
        prompts (list): List of prompts.
        prompt_path (str): The path to the prompt file.

    Returns:
        list: Ranked prompts.
    """
    try:
        validation_dataset = load_validation_dataset('test_dataset/test-data.json')
        responses = {prompt: generate_response(prompt, embed_model, index, prompt_path) for prompt in prompts}
        scores = {}

        for prompt in prompts:
            total_score = 0
            for data in validation_dataset:
                expected_responses = [data['assistant']]
                generated_response = responses[prompt]
                score = evaluate_response(prompt, generated_response, expected_responses)
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
    except Exception as e:
        logger.error(f"Error ranking prompts: {e}")
        raise

if __name__ == "__main__":
    prompts = [
        "1. Who are the team tutors for this week's challenge in Automatic Prompt Engineering?",
        '2. Which individuals are responsible for tutoring participants struggling with prompt engineering concepts?',
        '3. Who are the designated tutors for providing guidance and support in prompt optimization strategies?',
        '4. Who are the mentors available to assist with understanding prompt engineering tools and concepts?',
        '5. Who are the experts assigned to help others in the community with prompt engineering tasks?'
    ]
    prompt_path = 'prompts/response.txt'
    answer = ranking_prompts(prompts, prompt_path)
    print(answer)
