import streamlit as st
import json
from PyPDF2 import PdfReader
from io import StringIO

from scripts.data_generation import get_completion, file_reader
from scripts.generate_test_data import generate_test_data
from scripts.prompt_generation import generation_prompt

def read_file(uploaded_file):
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
    st.title("AI-Powered Dataset & Prompt Generator")

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
                prompt = file_reader("prompts/data-generation-prompt.txt")
                data_response = generate_test_data(prompt, context, num_test_output)
                test_data = data_response.content
                json_object = json.loads(test_data)
                
                json_filename = 'test_dataset/test-data.json'
                with open(json_filename, 'w') as f:
                    json.dump(json_object, f, indent=4)
                
                st.sidebar.success(f"JSON data has been saved as {json_filename}")
                
                st.session_state.json_object = json_object  # Save the JSON object in the session state
                
                prompt_response = generation_prompt(context, user_query, n=num_prompt_output)
                st.session_state.prompts = prompt_response.content  # Save the prompts in the session state

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
