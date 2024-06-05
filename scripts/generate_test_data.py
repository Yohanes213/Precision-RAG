from data_generation import file_reader, get_completion

def generate_test_data(prompt, context, num_test_output):

    response = get_completion(
        messages=[
            {
                "role": "user",
                "content": prompt.replace("{context}", context).replace("{num_test_output}", num_test_output)
            }
        ],
        model = 'davinci',
        logprobs=True,
        top_logprobs=1
    )

    return response

 