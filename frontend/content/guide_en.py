"""
English user-guide copy.

Generated from the copy that used to live inline in `app_pages/guide.py`: the
text is data now, so the page is a loop instead of two 270-line functions.
Each block is `(kind, payload)` where kind is one of:

* ``"markdown"``  — body passed to ``st.markdown``
* ``"html"``      — body passed to ``st.markdown(..., unsafe_allow_html=True)``
* ``"divider"``   — a horizontal rule
* ``"expander"``  — ``{"title", "expanded", "body"}`` rendered as an expander
"""
from __future__ import annotations

BLOCKS = [
    ('divider', None),
    ('markdown',
 '\n'
 '    ### Welcome to the IR System! \n'
 '    \n'
 '    This guide will walk you through using the Information Retrieval System step by step.\n'
 '    The system provides a complete workflow for document indexing, search, and evaluation with 8 main '
 'steps.\n'
 '    '),
    ('expander',
 {'title': '📋 **Overview**',
  'expanded': True,
  'body': '\n'
          '        The IR System is a comprehensive information retrieval platform featuring:\n'
          '        - Multiple indexing strategies (TF-IDF, BM25, Embeddings)\n'
          '        - Hybrid retrieval methods (Serial, Parallel Fusion)\n'
          '        - Query refinement techniques\n'
          '        - Comprehensive evaluation metrics\n'
          '        - Real-time system monitoring\n'
          '        '}),
    ('expander',
 {'title': '**1️⃣ Load Dataset** 📊',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Load a standard IR dataset for indexing and retrieval\n'
          '        \n'
          '        **Steps**:\n'
          '        1. Navigate to the **Dataset** page from the sidebar\n'
          '        2. Select a dataset from the dropdown (e.g., `beir/nfcorpus`, `cranfield`)\n'
          '        3. Click **Load Dataset**\n'
          '        4. Wait for confirmation that documents, queries, and qrels are loaded\n'
          '        \n'
          '        **Tips**:\n'
          '        - Start with smaller datasets like `cranfield` (~1,400 docs)\n'
          '        - `beir/nfcorpus` is good for medical domain (~3,600 docs)\n'
          '        - Large datasets like `msmarco-passage` may take significant time\n'
          '        '}),
    ('expander',
 {'title': '**2️⃣ Upload to Database** 💾',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Persist loaded data to PostgreSQL for efficient retrieval\n'
          '        \n'
          '        **Steps**:\n'
          '        1. After loading a dataset, click **Upload to Database**\n'
          '        2. Monitor the upload progress\n'
          '        3. Verify upload completion\n'
          '        \n'
          '        **Why?**: Database storage enables:\n'
          '        - Fast document retrieval\n'
          '        - Efficient query processing\n'
          '        - Persistent storage across sessions\n'
          '        '}),
    ('expander',
 {'title': '**3️⃣ Preprocess Documents** ⚙️',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Prepare text for indexing by normalizing and transforming it\n'
          '        \n'
          '        **Configuration Options**:\n'
          '        - **Tokenization**: spaCy (advanced) or Simple (fast)\n'
          '        - **Lowercase**: Convert all text to lowercase\n'
          '        - **Remove Punctuation**: Strip punctuation marks\n'
          '        - **Remove Stopwords**: Filter common words (the, is, a, etc.)\n'
          '        - **Morphological Processing**:\n'
          '          - **Lemmatization**: Convert words to dictionary form (learning → learn)\n'
          '          - **Stemming**: Aggressive suffix removal (machine → machin)\n'
          '          - **Neither**: Keep original word forms\n'
          '        - **Token Length**: Min/max character limits\n'
          '        \n'
          '        **Quick Presets**:\n'
          '        - **Default (Academic)**: Balanced for research datasets\n'
          '        - **Aggressive (Stemming)**: Maximum term reduction\n'
          '        - **Minimal**: Preserves most original terms\n'
          '        \n'
          '        **Testing**: Use the **Preprocessing Lab** to test configurations before indexing\n'
          '        '}),
    ('expander',
 {'title': '**4️⃣ Build Index** 🗂️',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Create searchable indexes from preprocessed documents\n'
          '        \n'
          '        **Index Types**:\n'
          '        \n'
          '        **TF-IDF (Vector Space Model)**:\n'
          '        - Classic term frequency-inverse document frequency\n'
          '        - Good baseline method\n'
          '        - Built automatically with inverted index\n'
          '        \n'
          '        **BM25**:\n'
          '        - Probabilistic retrieval model\n'
          '        - Parameters: k1 (term saturation), b (length normalization)\n'
          '        - Typically superior to TF-IDF\n'
          '        - Built from inverted index structure\n'
          '        \n'
          '        **Embeddings (Vector Index)**:\n'
          '        - Dense vector representations\n'
          '        - Enables semantic similarity search\n'
          '        - Uses `sentence-transformers/all-MiniLM-L6-v2` model\n'
          '        - Ideal for meaning-based queries\n'
          '        \n'
          '        **Steps**:\n'
          '        1. Go to **Indexing** page\n'
          '        2. Select preprocessing configuration\n'
          '        3. Choose index type(s) to build\n'
          '        4. Click **Build Index**\n'
          '        5. Monitor build progress\n'
          '        \n'
          '        **Build Times** (approximate):\n'
          '        - TF-IDF/BM25: ~2-3 minutes\n'
          '        - Embeddings: ~10-20 minutes (GPU recommended for large datasets)\n'
          '        \n'
          '        **Note**: You can build multiple indexes and compare their performance\n'
          '        '}),
    ('expander',
 {'title': '**5️⃣ Perform Search** 🔍',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Retrieve relevant documents for a query\n'
          '        \n'
          '        **Retrieval Methods**:\n'
          '        \n'
          '        **TF-IDF (Vector Space Model)**:\n'
          '        - Classic term-frequency based retrieval\n'
          '        - Good baseline method\n'
          '        \n'
          '        **BM25**:\n'
          '        - Probabilistic retrieval model\n'
          '        - Often outperforms TF-IDF\n'
          '        - Tune k1 (1.2-2.0) and b (0.75)\n'
          '        \n'
          '        **Embedding**:\n'
          '        - Semantic similarity using dense vectors\n'
          '        - Understands meaning beyond keywords\n'
          '        - Good for paraphrase queries\n'
          '        \n'
          '        **Hybrid Serial** (Re-ranking Pipeline):\n'
          '        1. First retriever gets candidate set (e.g., 1000 docs)\n'
          '        2. Second retriever re-ranks candidates\n'
          '        3. Returns top-k final results\n'
          '        \n'
          '        **Hybrid Parallel** (Fusion):\n'
          '        - Retrieves from multiple methods simultaneously\n'
          '        - Combines scores using fusion strategies:\n'
          '          - **RRF**: Reciprocal Rank Fusion\n'
          '          - **CombSUM**: Sum of normalized scores\n'
          '          - **CombMNZ**: CombSUM × number of methods\n'
          '          - **Weighted**: Custom weights per method\n'
          '        \n'
          '        **Steps**:\n'
          '        1. Navigate to **Search** page\n'
          '        2. Enter your query\n'
          '        3. Select retrieval method\n'
          '        4. Configure parameters\n'
          '        5. Click **Search**\n'
          '        6. Review ranked results\n'
          '        '}),
    ('expander',
 {'title': '**6️⃣ Refine Queries** 🔬 (Optional)',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Improve query quality through automatic enhancement\n'
          '        \n'
          '        **Techniques**:\n'
          '        \n'
          '        **Spell Correction**:\n'
          '        - Fixes typos automatically\n'
          '        - Configurable edit distance\n'
          '        \n'
          '        **Synonym Expansion**:\n'
          '        - Adds related terms from WordNet\n'
          '        - Configurable max synonyms per term\n'
          '        \n'
          '        **Query Suggestions**:\n'
          '        - Generates alternative phrasings\n'
          '        - Based on semantic similarity\n'
          '        \n'
          '        **Relevance Feedback**:\n'
          '        - Learns from past queries\n'
          '        - Suggests similar historical queries\n'
          '        \n'
          '        **Configuration**:\n'
          '        - Navigate to **Query Refinement** page\n'
          '        - Adjust parameters:\n'
          '          - Max edit distance (1-3)\n'
          '          - Confidence threshold (0-1)\n'
          '          - Max synonyms (0-5)\n'
          '          - History similarity threshold\n'
          '        '}),
    ('expander',
 {'title': '**7️⃣ Evaluate Performance** 📈',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Measure retrieval quality using standard IR metrics\n'
          '        \n'
          '        **Metrics**:\n'
          '        \n'
          '        **MAP (Mean Average Precision)**:\n'
          '        - Overall ranking quality\n'
          '        - Higher is better (0-1)\n'
          '        \n'
          '        **Recall**:\n'
          '        - Proportion of relevant docs retrieved\n'
          '        - Higher is better (0-1)\n'
          '        \n'
          '        **Precision@10**:\n'
          '        - Precision in top 10 results\n'
          '        - Higher is better (0-1)\n'
          '        \n'
          '        **nDCG (Normalized Discounted Cumulative Gain)**:\n'
          '        - Ranking quality with graded relevance\n'
          '        - Higher is better (0-1)\n'
          '        \n'
          '        **Steps**:\n'
          '        1. Go to **Evaluation** page\n'
          '        2. Select retrieval methods to compare\n'
          '        3. Configure eval parameters (cutoff, queries)\n'
          '        4. Click **Run Evaluation**\n'
          '        5. Review metric tables and charts\n'
          '        6. Compare methods\n'
          '        \n'
          '        **Tips**:\n'
          '        - Run evaluation on a subset of queries first\n'
          '        - BM25 typically outperforms TF-IDF\n'
          '        - Hybrid methods often achieve best results\n'
          '        - Embeddings excel at semantic queries\n'
          '        '}),
    ('expander',
 {'title': '**8️⃣ Monitor System Status** 🖥️',
  'expanded': False,
  'body': '\n'
          '        **Purpose**: Ensure all microservices are healthy and responsive\n'
          '        \n'
          '        **Services**:\n'
          '        - **API Gateway**: Main entry point\n'
          '        - **Preprocessing**: Text processing service\n'
          '        - **Indexing**: Index building service\n'
          '        - **Retrieval**: Search execution service\n'
          '        - **Ranking**: Re-ranking service\n'
          '        - **Query Refinement**: Query enhancement service\n'
          '        - **Evaluation**: Metrics computation service\n'
          '        \n'
          '        **Features**:\n'
          '        - Real-time health checks\n'
          '        - Response time monitoring\n'
          '        - Auto-refresh option\n'
          '        - Service dependency graph\n'
          '        \n'
          '        **Troubleshooting**:\n'
          '        - Red status: Service down or unreachable\n'
          '        - Yellow status: Slow response\n'
          '        - Green status: Healthy\n'
          '        \n'
          '        Navigate to **System Status** page to monitor.\n'
          '        '}),
    ('divider', None),
    ('expander',
 {'title': '💡 **Quick Tips**',
  'expanded': False,
  'body': '\n'
          '        - **Start Small**: Use `cranfield` or `beir/nfcorpus` for learning\n'
          '        - **Test Preprocessing**: Use Preprocessing Lab before building indexes\n'
          '        - **Compare Methods**: Evaluate multiple retrieval strategies\n'
          '        - **Monitor System**: Check System Status if operations fail\n'
          '        - **Save Configurations**: Document which configs work best for your use case\n'
          '        - **Hybrid Power**: Hybrid methods often outperform single methods\n'
          '        '}),
    ('expander',
 {'title': '🏗️ **System Architecture**',
  'expanded': False,
  'body': '\n'
          '        The IR System uses a **Service-Oriented Architecture (SOA)** with:\n'
          '        - **7 Microservices**: Each handles a specific task\n'
          '        - **API Gateway**: Routes requests to appropriate services\n'
          '        - **PostgreSQL Database**: Stores documents and metadata\n'
          '        - **Streamlit Frontend**: User interface\n'
          '        - **Docker**: Containerized deployment\n'
          '        \n'
          '        All services communicate via REST APIs for modularity and scalability.\n'
          '        '}),
]
