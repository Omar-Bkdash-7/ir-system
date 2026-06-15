FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm -q && \
    python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"
