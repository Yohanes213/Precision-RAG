import streamlit as st
import json
from PyPDF2 import PdfReader
from io import StringIO

from scripts.data_generation import AIAssistant, FileHandler
from scripts.generate_test_data import generate_test_data
from scripts.prompt_generation import generation_prompt
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import HumanMessage
from pinecone import Pinecone as PineconeClient
from scripts.prompt_ranking import ranking_prompts
from scripts.data_evaluation_pipeline import evaluation

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openapi_key = os.getenv('OPENAI_API_KEY')

# Initialize AIAssistant
assistant = AIAssistant(openai_api_key=openapi_key)

# Initialize embedding model and Pinecone client
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openapi_key)
pc = PineconeClient(pinecone_api_key)

index_name = 'prompt-engineering'
index = pc.Index(index_name)

def read_file(uploaded_file):
    """
    Read the uploaded file and extract text content.
    
    Args:
        uploaded_file (FileUploader): The file uploaded by the user.
    
    Returns:
        str: The text content extracted from the file.
    """
    if uploaded_file.type == "text/plain":
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.type == "application/pdf":
        content = ""
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            content += page.extract_text()
    else:
        st.error("Unsupported file format")
        return None
    return content

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Automatic Prompt Generator")

    # Initialize session state variables
    if 'json_object' not in st.session_state:
        st.session_state.json_object = None
    
    if 'prompts' not in st.session_state:
        st.session_state.prompts = None

    if 'view_json' not in st.session_state:
        st.session_state.view_json = False

    if 'view_prompts' not in st.session_state:
        st.session_state.view_prompts = False

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file", type=["text", "pdf"])
        user_query = st.text_input("Enter your message:")
        num_test_output = st.number_input("Number of test outputs", min_value=1, max_value=30, value=1)
        num_prompt_output = st.number_input("Number of possible prompts", min_value=1, max_value=30, value=1)

    if uploaded_file is not None:
        context = read_file(uploaded_file)
        if context:
            st.sidebar.success('File successfully read!')

            if st.sidebar.button("Generate Prompt"):
                # Generate test data
                prompt_data = FileHandler.read_file("prompts/data-generation-prompt.txt")
                data_response = generate_test_data(prompt_data, context, num_test_output)
                test_data = data_response.content
                json_object = json.loads(test_data)
                
                json_filename = 'test_dataset/test-data.json'
                with open(json_filename, 'w') as f:
                    json.dump(json_object, f, indent=4)

                # Evaluate the generated test data
                evaluation()
                
                st.sidebar.success(f"JSON data has been saved as {json_filename}")
                
                st.session_state.json_object = json_object  # Save the JSON object in the session state
                
                # Generate prompts based on user query and context
                prompt_path = 'prompts/prompt-generation.txt'
                prompt_response = generation_prompt(context, user_query, prompt_path, n=num_prompt_output)
                prompt_response_path = 'prompts/response.txt'

                # Rank the generated prompts
                ranked_prompts = ranking_prompts(list(prompt_response.content.split('\n')), prompt_response_path)

                st.session_state.prompts = ranked_prompts  # Save the prompts in the session state

    if st.sidebar.button("View the json file"):
        st.session_state.view_json = True  # Set a flag to view JSON content
        st.session_state.view_prompts = False  # Reset the prompt view flag

    if st.sidebar.button("View Prompts"):
        st.session_state.view_prompts = True  # Set a flag to view prompts
        st.session_state.view_json = False  # Reset the JSON view flag

    # Display the JSON content if the flag is set
    if st.session_state.view_json and st.session_state.json_object:
        st.json(st.session_state.json_object)
        st.download_button(label="Download JSON", data=json.dumps(st.session_state.json_object, indent=4), file_name='test-data.json', mime="application/json")

    # Display the prompts if the flag is set
    if st.session_state.view_prompts and st.session_state.prompts:
        st.write(st.session_state.prompts)

if __name__ == "__main__":
    main()
