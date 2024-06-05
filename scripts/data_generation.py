import os
import json
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(openai_api_key)

def get_completion(
                    model,
                    messages, 
                    max_tokens=500, 
                    temperature=0, 
                    stop=None,
                    seed=123,
                    tools=None,
                    logprobs=None,
                    top_logprobs=None):
    
    params = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stop': stop,
        'seed': seed,
        'logprobs': logprobs,
        'top_logprobs': top_logprobs
    }

    if tools:
        params['tools'] = tools
    completion = client.chat.completion.create(**params)

    return completion