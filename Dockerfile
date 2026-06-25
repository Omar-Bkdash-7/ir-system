FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install psycopg2-binary==2.9.9
RUN python -m spacy download en_core_web_sm -q

ENV NLTK_DATA=/root/nltk_data
RUN mkdir -p /root/nltk_data && \
    python -c "import nltk; \
    nltk.download('wordnet', download_dir='/root/nltk_data', quiet=False); \
    nltk.download('stopwords', download_dir='/root/nltk_data', quiet=False); \
    nltk.download('omw-1.4', download_dir='/root/nltk_data', quiet=False); \
    nltk.download('punkt', download_dir='/root/nltk_data', quiet=False); \
    print('NLTK data downloaded successfully')"
