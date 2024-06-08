# Precision Rag

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

Precision RAG is a project focused on building enterprise-grade Retrieval-Augmented Generation (RAG) systems with a strong emphasis on prompt tuning. This repository provides tools and scripts to facilitate data generation, prompt tuning, evaluation, and deployment of RAG systems.

## Table of Contents
- [Project Structure](#project-structure)

- [Setup Instruction](#setup-instruction)

- [Running the Application](#running-the-application)

- [Demo](#demo)

- [Contributing](#contributing)

- [License](#license)

## Project Structure
- **.github/workflows:** Stores the workflow script (precision-rag-cl.yml) for automated CI builds using GitHub Actions.
- **logs:** Contains a log file (logs.log) to track project execution.
- **prompts:** Houses various prompt files used for different stages:
  - **context.txt:** Defines prompts for providing context to the model.
  - **data-generation.txt:** Stores prompts for data generation.
  - **prompt-generation.txt:** Contains prompts used for generating prompts themselves (meta-prompting).
  - **response.txt:** Defines prompts for response generation.
  - **validation.txt:** Stores prompts for validation purposes.
- **scripts:** Contains Python scripts for various functionalities:
  - **data_evaluation_pipeline.py:** Script for evaluating generated data.
  - **data_generation.py:** Script for generating training or testing data.
  - **generate_test_data.py:** Script for generating specific test data (if needed).
  - **prompt_generation.py:** Script for generating prompts based on meta-prompting techniques.
  - **prompt_ranking.py:** Script for ranking retrieved passages or documents.
  - **rag.py:** Core script implementing the RAG system with prompt tuning.
- **test_dataset:** Contains a test data file (test_data.json) for evaluation purposes.
- **.gitignore:** Specifies files and directories to be excluded from version control.
- **readme.md:** The current file you're reading (project documentation).
- **app.py:** Script for the Streamlit application providing a user interface (if applicable).
- **file.json:** Additional configuration file (modify as needed).
- **requirements.txt:** Lists the required Python libraries for running the project.


## Setup Instructions

**Prerequisites**

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key
- Required libraries listed in `requirements.txt`

**Installation**

1. Clone the repository:
    ```sh
    git clone https://github.com/Yohanes213/Precision-RAG
    cd Precision-RAG
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        ```
   - Add your Pinecone API key to the `.env` file:
        ```
        PINECONE_API_KEY=your_pinecone_api_key
        ```

## Running the Application

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Configure the settings in the sidebar and click "Generate Prompt".
   
4. Click on 'View Prompts' to view the prompt Ranking.

5. Click on `View the json file` to view and download the generated validation dataset.



## Demo

https://github.com/Yohanes213/Precision-RAG/assets/99422479/6e4f9432-3300-44de-9ef8-0ca6e44b0be4


## Contributing


Feel free to open issues or submit pull requests if you have any improvements or suggestions.

## License

This project is licensed under the MIT License.



