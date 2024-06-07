from data_generation import FileHandler, AIAssistant
import json
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
from pathlib import Path


# Load environment variables
env_path = Path('.env')
load_dotenv(env_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openai_api_key)


def generate_test_data(prompt, context, num_test_output):
    
    response = assistant.get_chat_completion(
        #model='davinci',

        messages=[
            HumanMessage(content= prompt.replace("{context}", context).replace("{num_test_output}", str(num_test_output)))
            ],
        logprobs=True,
        top_logprobs=1
    )

    return response



def main(num_test_output):
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
    
    print(f"JSON data has been saved.")


if __name__ == "__main__":
    main(10)

 