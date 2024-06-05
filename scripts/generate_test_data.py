from data_generation import file_reader, get_completion
import json
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)


def _convert_message_to_dict(message):
  if isinstance(message, str):
    return {"content": message}  # Handle strings directly
  else:
    return {"content": message.content}  # Use content attribute for objects




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
    context_message = file_reader("prompts/context.txt")
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

 