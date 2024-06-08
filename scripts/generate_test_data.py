import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from scripts.data_generation import FileHandler, AIAssistant

# Load environment variables
env_path = Path('.env')
load_dotenv(env_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openai_api_key)

def generate_test_data(prompt, context, num_test_output):
    """
    Generate test data using the provided prompt and context.

    Args:
        prompt (str): The prompt template.
        context (str): The context to insert into the prompt.
        num_test_output (int): The number of test outputs to generate.

    Returns:
        response: The API response containing the generated test data.

    Raises:
        Exception: If there is an error in generating test data.
    """
    try:
        response = assistant.get_chat_completion(
            messages=[
                HumanMessage(content=prompt.replace("{context}", context).replace("{num_test_output}", str(num_test_output)))
            ],
            logprobs=True,
            top_logprobs=1
        )
        logger.info("Test data generation successful.")
        return response
    except Exception as e:
        logger.error(f"Error in generating test data: {e}")
        raise

def main(num_test_output):
    """
    Main function to generate test data and save it to a JSON file.

    Args:
        num_test_output (int): The number of test outputs to generate.

    Raises:
        Exception: If there is an error in the main function.
    """
    try:
        context_message = FileHandler.read_file("prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf")
        prompt_message = FileHandler.read_file("prompts/data-generation-prompt.txt")

        context = str(context_message)
        prompt = str(prompt_message)

        response = generate_test_data(
            prompt=prompt,
            context=context,
            num_test_output=num_test_output
        )
        
        test_data = response.content
        json_object = json.loads(test_data)
        with open('test_dataset/test-data.json', 'w') as f:
            json.dump(json_object, f, indent=4)
        
        logger.info("JSON data has been saved successfully.")
        print("JSON data has been saved.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main(10)
