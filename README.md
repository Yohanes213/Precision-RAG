# Precision Rag

Precision RAG is a project focused on building enterprise-grade Retrieval-Augmented Generation (RAG) systems with a strong emphasis on prompt tuning. This repository provides tools and scripts to facilitate data generation, prompt tuning, evaluation, and deployment of RAG systems.

## Project Structure

.
├── .github/
│   └── workflows/
│       └── precision.py
├── logs/
│   └── logs.log
├── prompts/
│   ├── context.txt
│   ├── data-generation.txt
│   ├── prompt-generation.txt
│   ├── response.txt
│   └── validation.txt
├── scripts/
│   ├── data_evaluation_pipeline.py
│   ├── data_generation.py
│   ├── generate_test_data.py
│   ├── prompt_generation.py
│   ├── prompt_ranking.py
│   └── rag.py
├── test_dataset/
│   └── test_data.json
├── .gitignore
├── readme.md
├── app.py
├── file.json
└── requirements.txt

## Project Structure

### `scripts/`
- `data_generation.py`: Contains functions to interact with OpenAI's API and read files.
- `generate_test_data.py`: Utilizes the functions in `data_generation.py` to generate test data.

### `app.py`
- Streamlit application that provides a user-friendly interface for generating test data.

### `prompts/`
- `10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf`: PDF file used as context.
- `data-generation-prompt.txt`: Text file containing the prompt for data generation.

### `test_dataset/`
- Directory where generated test data in JSON format will be saved.

### `logs/`
- Directory for log files.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Streamlit

### Installation

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

## Running the Application

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Configure the settings in the sidebar and click "Generate Test Data".

## Usage

- **Context File Path**: Path to the context file (e.g., `prompts/10 Academy Cohort B - Weekly Challenge_ Week - 7.pdf`).
- **Prompt File Path**: Path to the prompt file (e.g., `prompts/data-generation-prompt.txt`).
- **Number of Test Outputs**: Number of test outputs to generate.

The generated test data will be displayed on the web interface and saved to `test_dataset/test-data.json`.

## Demo

https://github.com/Yohanes213/Precision-RAG/assets/99422479/6e4f9432-3300-44de-9ef8-0ca6e44b0be4


## Contributing


Feel free to open issues or submit pull requests if you have any improvements or suggestions.

## License

This project is licensed under the MIT License.



