from scripts.data_generation import file_reader, get_completion
import json
from langchain.schema import HumanMessage
  


def generate_test_data(prompt, context, num_test_output):
    response = get_completion(
        #model='davinci',

        messages=[
            HumanMessage(content= prompt.replace("{context}", context).replace("{num_test_output}", str(num_test_output)))
            ],
        logprobs=True,
        top_logprobs=1
    )

    return response



def main(num_test_output):
    context_message = file_reader("prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf")
    prompt_message = file_reader("prompts/data-generation-prompt.txt")

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
    main(5)

 