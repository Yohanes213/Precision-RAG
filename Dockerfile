FROM python:3.8-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt.

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY..

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run command to start Streamlit app
CMD ["streamlit", "run", "app.py"]