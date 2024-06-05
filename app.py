import streamlit as st
import json
from PyPDF2 import PdfReader
from io import StringIO

from scripts.data_generation import get_completion, file_reader
from scripts.generate_test_data import generate_test_data


# def read_file(uploaded_file):
#     if uploaded_file.type =

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
    st.title("Validation Dataset Generator")

    uploaded_file = st.file_uploader("Choose a file", type=["text", "pdf"])

    if uploaded_file is not None:
        context = read_file(uploaded_file)
        if context:
            st.write('File successfully read!')

            num_test_output = st.number_input("Number of test outputs", min_value = 1, max_value=30, value=1)

            if st.button("Generate JSON"):
                prompt = file_reader("prompts/data-generation-prompt.txt")

                response = generate_test_data(prompt, context, num_test_output)

                test_data = response.content
                json_object = json.loads(test_data)

                json_filename = 'test_dataset/test-data.json'
                with open(json_filename, 'w') as f:
                    json.dump(json_object, f, indent=4)
                
                st.success(f"JSON data has been saved as {json_filename}")
                st.download_button(label="Download JSON", data=json.dumps(json_object, indent=4), file_name=json_filename, mime="application/json")


if __name__ == "__main__":
    main()




