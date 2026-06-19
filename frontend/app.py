import streamlit as st
import requests
import time
import os
from typing import Dict, List, Optional

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api_gateway:8000")
INDEXING_URL = os.getenv("INDEXING_URL", "http://indexing:8002")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8003")
RANKING_URL = os.getenv("RANKING_URL", "http://ranking:8004")
QUERY_REFINEMENT_URL = os.getenv("QUERY_REFINEMENT_URL", "http://query_refinement:8005")
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://evaluation:8006")

st.set_page_config(page_title="IR System", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { padding: 0.5rem 0.5rem 0.5rem 1rem; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; padding-left: 0.8rem !important; padding-right: 1rem !important; max-width: 100%; }
    h1 { color: #a6e3a1 !important; font-size: 1.6rem !important; margin: 0 0 0.3rem 0 !important; font-weight: 700 !important; }
    h2 { color: #89b4fa !important; font-size: 1.2rem !important; margin: 0.5rem 0 !important; font-weight: 600 !important; }
    h3 { color: #cdd6f4 !important; font-size: 1rem !important; margin: 0.3rem 0 !important; font-weight: 600 !important; }
    h5 { color: #a6adc8 !important; font-size: 0.9rem !important; margin: 0 0 0.5rem 0 !important; font-weight: 500 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; padding: 0; margin-bottom: 0.8rem; }
    .stTabs [data-baseweb="tab"] { height: 32px; background: rgba(49,50,68,0.4); border-radius: 6px; padding: 0 12px; color: #cdd6f4; font-weight: 500; font-size: 0.85rem; border: 1px solid transparent; }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(69,71,90,0.6); border-color: #89b4fa; }
    .stTabs [aria-selected="true"] { background: #89b4fa !important; color: #1e1e2e !important; border-color: #89b4fa !important; font-weight: 600; }
    .stButton>button { background: linear-gradient(135deg,#89b4fa,#74c7ec); color: #1e1e2e; border: none; border-radius: 6px; padding: 0.35rem 0.7rem; font-weight: 600; font-size: 0.8rem; white-space: nowrap; width: 100%; }
    .stButton>button:hover { background: linear-gradient(135deg,#74c7ec,#89dceb); transform: translateY(-1px); }
    .stTextInput>div>div>input { background: rgba(49,50,68,0.6); border: 1px solid #45475a; border-radius: 6px; padding: 0.4rem 0.7rem; color: #cdd6f4; font-size: 0.9rem; }
    .stTextInput>div>div>input:focus { border-color: #89b4fa; box-shadow: 0 0 0 2px rgba(137,180,250,0.2); }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #89b4fa; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #a6adc8; }
    .result-card { background: linear-gradient(135deg,rgba(49,50,68,0.8),rgba(69,71,90,0.6)); border: 1px solid #45475a; border-radius: 10px; padding: 0.8rem; margin: 0.5rem 0; transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
    .result-card:hover { border-color: #a6e3a1; transform: translateY(-2px); box-shadow: 0 4px 16px rgba(166,227,161,0.2); }
    .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; padding-bottom: 0.4rem; border-bottom: 1px solid #45475a; }
    .doc-id { color: #89b4fa; font-weight: 600; font-size: 0.9rem; }
    .score-badge { background: linear-gradient(135deg,#a6e3a1,#94e2d5); color: #1e1e2e; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 700; font-size: 0.8rem; }
    .result-text { color: #cdd6f4; line-height: 1.5; font-size: 0.85rem; }
    .info-box, .success-box, .warning-box, .error-box { padding: 0.6rem 0.8rem; border-radius: 6px; margin: 0.5rem 0; font-size: 0.85rem; }
    .info-box { background: rgba(137,180,250,0.15); border-left: 3px solid #89b4fa; color: #cdd6f4; }
    .success-box { background: rgba(166,227,161,0.15); border-left: 3px solid #a6e3a1; color: #a6e3a1; }
    .warning-box { background: rgba(249,226,175,0.15); border-left: 3px solid #f9e2af; color: #f9e2af; }
    .error-box { background: rgba(243,139,168,0.15); border-left: 3px solid #f38ba8; color: #f38ba8; }
    hr { border: none; border-top: 1px solid #45475a; margin: 0.8rem 0; }
    section[data-testid="stSidebar"]>div { padding-top: 0rem; }
    section[data-testid="stSidebar"] h2 { font-size: 1rem !important; margin: -0.8rem 0 0.3rem 0 !important; }
    section[data-testid="stSidebar"] button[kind="header"] { display: none; }
    .stExpander { background: rgba(49,50,68,0.3); border: 1px solid #45475a; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

load_css()

def safe_request(method: str, url: str, **kwargs) -> Optional[Dict]:
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_dataset_info(): return safe_request("GET", f"{API_GATEWAY_URL}/datasets/info")
def get_tfidf_status(): return safe_request("GET", f"{RETRIEVAL_URL}/status/tfidf")
def get_embedding_status(): return safe_request("GET", f"{RETRIEVAL_URL}/status/embedding")
def get_bm25_status(): return safe_request("GET", f"{RETRIEVAL_URL}/status/bm25")
def build_tfidf_index(force=False): return safe_request("POST", f"{INDEXING_URL}/index/tfidf/build?force_rebuild={force}")
def build_embedding_index(force=False): return safe_request("POST", f"{INDEXING_URL}/index/embedding/build?force_rebuild={force}")
def build_bm25_index(force=False): return safe_request("POST", f"{INDEXING_URL}/index/bm25/build?force_rebuild={force}")

def render_result_card(doc: Dict, idx: int):
    doc_id = doc.get("doc_id", "N/A")
    score = doc.get("score", 0.0)
    text = doc.get("text", "")
    snippet = text[:350] + "..." if len(text) > 350 else text
    st.markdown(f"""
    <div class="result-card">
        <div class="result-header">
            <span class="doc-id">📄 Doc #{idx} • {doc_id}</span>
            <span class="score-badge">⭐ {score:.4f}</span>
        </div>
        <div class="result-text">{snippet}</div>
    </div>
    """, unsafe_allow_html=True)

st.title("🔍 Advanced IR System")
st.markdown("##### Multi-Model Document Retrieval System")

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    dataset_info = get_dataset_info()
    if dataset_info:
        st.markdown(f"<div class='success-box'>✅ {dataset_info.get('dataset_name','N/A')}</div>", unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: st.metric("Docs",f"{dataset_info.get('doc_count',0):,}")
        with col2: st.metric("Queries",f"{dataset_info.get('query_count',0):,}")
    else:
        st.markdown("<div class='error-box'>❌ Not loaded</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🛠️ Indexes")
    tfidf_status = get_tfidf_status()
    embedding_status = get_embedding_status()
    bm25_status = get_bm25_status()
    
    all_ready = tfidf_status and embedding_status and bm25_status and embedding_status.get('status')=='ready' and bm25_status.get('status')=='ready'
    
    if tfidf_status:
        st.markdown("✅ **TF-IDF**: Ready")
        st.caption(f"{tfidf_status.get('total_docs',0):,} documents")
    else:
        st.markdown("⏳ **TF-IDF**: Loading...")
    
    if embedding_status and embedding_status.get('status')=='ready':
        st.markdown("✅ **Embedding**: Ready")
        st.caption(f"{embedding_status.get('total_docs',0):,} documents")
    else:
        st.markdown("⏳ **Embedding**: Loading...")
    
    if bm25_status and bm25_status.get('status')=='ready':
        st.markdown("✅ **BM25**: Ready")
        st.caption(f"{bm25_status.get('total_docs',0):,} documents")
    else:
        st.markdown("⏳ **BM25**: Loading...")
    
    if not all_ready:
        st.info("Indexes are loading in the background...")
        time.sleep(2)
        st.rerun()
    st.markdown("---")
    enhanced_mode = st.toggle("🟢 Enhanced", help="Query Refinement + RAG")
    if enhanced_mode:
        st.markdown("<div class='info-box'>Coming soon</div>", unsafe_allow_html=True)

query = st.text_input("🔎 Search", placeholder="Enter your query...", help="Type your question")
st.markdown("---")

st.markdown("### 🎯 Select Retrieval Models")
col1, col2, col3, col4 = st.columns(4)
with col1:
    use_tfidf = st.checkbox("📊 TF-IDF", value=True, key="use_tfidf")
with col2:
    use_bm25 = st.checkbox("📈 BM25", value=True, key="use_bm25")
with col3:
    use_embedding = st.checkbox("🧠 Embedding", value=True, key="use_embedding")
with col4:
    use_hybrid = st.checkbox("🔀 Hybrid", value=False, key="use_hybrid", help="⚠️ In progress")

st.markdown("---")

selected_models = []
if use_tfidf: selected_models.append("tfidf")
if use_embedding: selected_models.append("embedding")
if use_bm25: selected_models.append("bm25")

if use_tfidf:
    with st.expander("📊 TF-IDF Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tfidf_k = st.slider("Candidates to Retrieve", 10, 200, 100, 5, key="tfidf_k", help="Internal retrieval size")
        with col2:
            tfidf_disp = st.slider("Results to Display", 5, 50, 20, 5, key="tfidf_disp", help="Shown in UI")

if use_embedding:
    with st.expander("🧠 Embedding Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            emb_k = st.slider("Candidates to Retrieve", 10, 200, 100, 5, key="emb_k", help="Internal retrieval size")
        with col2:
            emb_disp = st.slider("Results to Display", 5, 50, 20, 5, key="emb_disp", help="Shown in UI")
        st.info("🧠 Model: all-MiniLM-L6-v2 (384 dimensions)")

if use_bm25:
    with st.expander("📈 BM25 Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            bm25_k = st.slider("Candidates to Retrieve", 10, 200, 100, 5, key="bm25_k", help="Internal retrieval size")
            bm25_k1 = st.slider("k1 (Term Saturation)", 0.0, 3.0, 1.5, 0.1, key="bm25_k1", help="Controls term frequency saturation")
        with col2:
            bm25_disp = st.slider("Results to Display", 5, 50, 20, 5, key="bm25_disp", help="Shown in UI")
            bm25_b = st.slider("b (Length Normalization)", 0.0, 1.0, 0.75, 0.05, key="bm25_b", help="Document length normalization")
        st.latex(r"BM25(q,d) = \sum_{t \in q} IDF(t) \cdot \frac{TF(t,d) \cdot (k_1 + 1)}{TF(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}")
        st.info(f"📊 Current parameters: k1={bm25_k1:.1f}, b={bm25_b:.2f}")

if use_hybrid:
    with st.expander("🔀 Hybrid Configuration", expanded=True):
        st.markdown("<div class='warning-box'>⚠️ Hybrid modes in progress</div>", unsafe_allow_html=True)
        hybrid_type = st.radio("Strategy", ["Serial (Re-ranking)", "Parallel (Fusion)"], key="hybrid_type")
        
        if "Serial" in hybrid_type:
            st.markdown("**Serial Pipeline**")
            col1, col2 = st.columns(2)
            with col1:
                first_stage = st.selectbox("1st Stage (Fast)", ["BM25", "TF-IDF"], help="Initial retrieval")
            with col2:
                second_stage = st.selectbox("2nd Stage (Strong)", ["Embedding", "TF-IDF"], help="Re-ranking")
            st.info(f"📊 Pipeline: {first_stage} (1000 candidates) → {second_stage} (Top 50)")
        else:
            st.markdown("**Parallel Fusion**")
            fusion_method = st.selectbox("Fusion Method", ["CombSUM", "CombMNZ", "Weighted Average"])
            if fusion_method == "Weighted Average":
                st.markdown("**Model Weights**")
                if use_tfidf:
                    tfidf_weight = st.slider("TF-IDF Weight", 0.0, 1.0, 0.33, 0.01)
                if use_embedding:
                    emb_weight = st.slider("Embedding Weight", 0.0, 1.0, 0.33, 0.01)
                if use_bm25:
                    bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.34, 0.01)

st.markdown("---")
if st.button("🚀 Search", type="primary"):
    if not query or not query.strip():
        st.error("❌ Enter a query")
    elif not selected_models and not use_hybrid:
        st.error("❌ Select at least one model")
    else:
        start = time.time()
        with st.spinner("🔄 Searching..."):
            all_results = []
            for model in selected_models:
                if model == "tfidf":
                    top_k = tfidf_k
                    display_k = tfidf_disp
                elif model == "embedding":
                    top_k = emb_k
                    display_k = emb_disp
                elif model == "bm25":
                    top_k = bm25_k
                    display_k = bm25_disp
                else:
                    continue
                
                req_data = {
                    "query_text": query,
                    "query_tokens": query.lower().split(),
                    "model": model,
                    "top_k": top_k,
                    "include_text": True
                }
                
                if model == "bm25":
                    req_data["bm25_k1"] = bm25_k1
                    req_data["bm25_b"] = bm25_b
                
                res = safe_request("POST", f"{RETRIEVAL_URL}/retrieve/{model}", json=req_data)
                if res:
                    all_results.append({"model": model, "results": res.get("results", [])[:display_k]})
            
            elapsed = (time.time()-start)*1000
            if all_results:
                st.markdown("---")
                st.markdown("## 📊 Results")
                col1,col2,col3=st.columns(3)
                with col1: st.metric("⏱️ Time",f"{elapsed:.0f} ms")
                with col2: st.metric("📄 Docs",sum(len(r["results"]) for r in all_results))
                with col3: st.metric("🎯 Models",len(all_results))
                st.markdown("---")
                if enhanced_mode:
                    st.markdown("### 🤖 AI Answer")
                    st.markdown("<div class='info-box'><b>RAG Preview:</b> AI-generated answers will appear here using Flan-T5.</div>", unsafe_allow_html=True)
                    st.markdown("---")
                for mr in all_results:
                    st.markdown(f"### 📊 {mr['model'].upper()} ({len(mr['results'])} docs)")
                    for idx,doc in enumerate(mr['results'],1):
                        render_result_card(doc,idx)
                    st.markdown("---")
            else:
                st.markdown("<div class='error-box'>❌ No results</div>", unsafe_allow_html=True)

st.markdown("---")
with st.expander("📈 Statistics"):
    col1,col2,col3,col4=st.columns(4)
    with col1: st.metric("Total Docs","522,931")
    with col2: st.metric("Vocab",f"{get_tfidf_status().get('vocab_size',0):,}" if get_tfidf_status() else "0")
    with col3: st.metric("Test Queries","10,000")
    with col4: st.metric("Qrels","15,675")
