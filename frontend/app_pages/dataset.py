"""
Dataset page: browser, download with progress, local list + delete, DB upload.
"""
from __future__ import annotations

import time
import streamlit as st

import config
import api.gateway as gateway
import api.indexing as indexing_api
from constants import SS_DATASET_VALIDATION_RESULT, SS_VALIDATION_IN_PROGRESS

# ---------------------------------------------------------------------------
# Curated catalogue (from ir-datasets.com/counts) — only rows with qrels
# Columns: (dataset_id, display_name, docs, queries, qrels, description)
# ---------------------------------------------------------------------------
CATALOGUE = [
    ("antique/test",                        "ANTIQUE / test",                   404_000,     200,    6_600, "Community QA – Yahoo Answers"),
    ("antique/train",                       "ANTIQUE / train",                  404_000,   2_400,   27_000, "Community QA – Yahoo Answers"),
    ("beir/arguana",                        "BEIR – ArguAna",                     8_700,   1_400,    1_400, "Argument retrieval"),
    ("beir/climate-fever",                  "BEIR – Climate-FEVER",           5_400_000,   1_500,    4_700, "Fact verification – climate"),
    ("beir/cqadupstack/android",            "BEIR – CQADupStack / android",      23_000,     699,    1_700, "StackExchange duplicate questions"),
    ("beir/cqadupstack/english",            "BEIR – CQADupStack / english",      40_000,   1_600,    3_800, "StackExchange duplicate questions"),
    ("beir/cqadupstack/gaming",             "BEIR – CQADupStack / gaming",       45_000,   1_600,    2_300, "StackExchange duplicate questions"),
    ("beir/cqadupstack/tex",                "BEIR – CQADupStack / tex",          68_000,   2_900,    5_200, "StackExchange duplicate questions"),
    ("beir/dbpedia-entity/test",            "BEIR – DBpedia Entity / test",   4_600_000,     400,   44_000, "Entity retrieval"),
    ("beir/fever/test",                     "BEIR – FEVER / test",            5_400_000,   6_700,    7_900, "Fact verification"),
    ("beir/fiqa/test",                      "BEIR – FiQA / test",                58_000,     648,    1_700, "Financial QA"),
    ("beir/hotpotqa/test",                  "BEIR – HotpotQA / test",         5_200_000,   7_400,   15_000, "Multi-hop QA"),
    ("beir/msmarco/dev",                    "BEIR – MSMARCO / dev",           8_800_000,   7_000,    7_400, "Web passage ranking"),
    ("beir/nfcorpus/test",                  "BEIR – NFCorpus / test",             3_600,     323,   12_000, "Medical IR – nutrition"),
    ("beir/nq",                             "BEIR – Natural Questions",       2_700_000,   3_500,    4_200, "Open-domain QA"),
    ("beir/quora/test",                     "BEIR – Quora / test",              523_000,  10_000,   16_000, "Duplicate question detection"),
    ("beir/scidocs",                        "BEIR – SciDocs",                    26_000,   1_000,   30_000, "Scientific document retrieval"),
    ("beir/scifact/test",                   "BEIR – SciFact / test",              5_200,     300,      339, "Scientific claim verification"),
    ("beir/trec-covid",                     "BEIR – TREC-COVID",                171_000,      50,   66_000, "COVID-19 biomedical IR"),
    ("beir/webis-touche2020",               "BEIR – Touché 2020",               383_000,      49,    3_000, "Argument retrieval"),
    ("codec",                               "CODEC",                            730_000,      42,    6_200, "Complex domain retrieval"),
    ("cranfield",                           "Cranfield",                          1_400,     225,    1_800, "Classic IR benchmark"),
    ("disks45/nocr/trec-robust-2004",       "TREC Robust04",                    528_000,     250,  311_000, "News IR"),
    ("lotte/lifestyle/test/search",         "LoTTE – Lifestyle / search",       119_000,     661,    1_800, "Long-tail topic retrieval"),
    ("lotte/recreation/test/search",        "LoTTE – Recreation / search",      167_000,     924,    2_000, "Long-tail topic retrieval"),
    ("lotte/technology/test/search",        "LoTTE – Technology / search",      639_000,     596,    2_000, "Long-tail topic retrieval"),
    ("lotte/writing/test/search",           "LoTTE – Writing / search",         200_000,   1_100,    3_500, "Long-tail topic retrieval"),
    ("msmarco-passage/dev/small",           "MSMARCO Passage / dev-small",    8_800_000,   7_000,    7_400, "Large-scale passage ranking"),
    ("msmarco-passage/trec-dl-2019/judged", "MSMARCO – TREC DL 2019",         8_800_000,      43,    9_300, "TREC Deep Learning 2019"),
    ("msmarco-passage/trec-dl-2020/judged", "MSMARCO – TREC DL 2020",         8_800_000,      54,   11_000, "TREC Deep Learning 2020"),
    ("msmarco-document/dev",                "MSMARCO Document / dev",          3_200_000,   5_200,    5_200, "Document ranking"),
    ("nfcorpus/test",                       "NFCorpus / test",                    5_400,     325,   16_000, "Medical nutrition IR"),
    ("vaswani",                             "Vaswani",                           11_000,      93,    2_100, "Classic IR collection"),
]

_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _parse_filter(v: str):
    if v == "Any":
        return None
    raw = v.replace(",", "").upper()
    for s, m in _MULT.items():
        if raw.endswith(s):
            return int(float(raw[:-1]) * m)
    return int(raw)


# ---------------------------------------------------------------------------
# Section: catalogue browser
# ---------------------------------------------------------------------------
def _trigger_filter():
    """on_change callback — forces an immediate rerun which re-evaluates the filter."""
    st.session_state["_filter_ts"] = time.time()


def _render_browser() -> str | None:
    st.markdown("### Browse Datasets")
    
    st.info("💡 **Note:** Showing datasets compatible with ir_datasets 0.5.9. Some newer datasets (like nano-beir variants) may require upgrading ir_datasets.")

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        st.text_input(
            "Search", placeholder="quora, medical, fever …",
            key="ds_search",
        )
    with c2:
        st.selectbox("Max docs", ["Any", "10K", "100K", "1M", "10M"], key="ds_max_docs")
    with c3:
        st.selectbox("Min queries", ["Any", "50", "200", "1K", "5K"], key="ds_min_q")
    with c4:
        st.selectbox("Min qrels", ["Any", "500", "1K", "5K", "10K"], key="ds_min_r")

    term     = st.session_state.get("ds_search", "").strip().lower()
    max_docs = _parse_filter(st.session_state.get("ds_max_docs", "Any"))
    min_q    = _parse_filter(st.session_state.get("ds_min_q",    "Any"))
    min_r    = _parse_filter(st.session_state.get("ds_min_r",    "Any"))

    # Filtering is reactive — runs on every rerun triggered by widget changes (no button needed)
    rows = [
        r for r in CATALOGUE
        if (not term or term in r[0] or term in r[1].lower() or term in r[5].lower())
        and (max_docs is None or r[2] <= max_docs)
        and (min_q    is None or r[3] >= min_q)
        and (min_r    is None or r[4] >= min_r)
    ]

    if not rows:
        st.info("No datasets match the current filters.")
        return None

    st.caption(f"{len(rows)} dataset(s) found — click a row to select it")

    hdr = st.columns([3, 1, 1, 1, 4])
    for col, lbl in zip(hdr, ["Dataset ID", "Docs", "Queries", "Qrels", "Description"]):
        col.markdown(f"**{lbl}**")

    selected = None
    for ds_id, name, docs, queries, qrels, desc in rows:
        cols = st.columns([3, 1, 1, 1, 4])
        with cols[0]:
            if st.button(ds_id, key=f"sel_{ds_id}", use_container_width=True):
                selected = ds_id
        cols[1].write(_fmt(docs))
        cols[2].write(_fmt(queries))
        cols[3].write(_fmt(qrels))
        cols[4].write(desc)

    return selected


# ---------------------------------------------------------------------------
# Section: download + validate with fake progress bar
# ---------------------------------------------------------------------------
def _render_download():
    st.markdown("### Download & Validate")
    st.caption(
        "Select a dataset above or type an ID. "
        "If already downloaded locally it is loaded instantly. "
        "Otherwise ir_datasets downloads it first."
    )

    col_id, col_btn = st.columns([3, 2])
    with col_id:
        dataset_id: str = st.text_input(
            "Dataset ID",
            value=st.session_state.get("ds_selected_id", config.DATASET_NAME),
            key="dataset_id_input",
        )
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        do_download = st.button("⬇️ Download & Validate", type="primary",
                                use_container_width=True, key="btn_dl")

    if not do_download:
        val = st.session_state.get(SS_DATASET_VALIDATION_RESULT)
        if val:
            if "error" not in val:
                st.success("Last validation succeeded.")
                with st.expander("Validation response", expanded=False):
                    st.json(val)
            else:
                st.error(f"Last validation failed: {val.get('error')}")
        return

    # ── Run download with animated progress ──────────────────────────────
    st.session_state[SS_VALIDATION_IN_PROGRESS] = True
    placeholder = st.empty()

    with placeholder.container():
        prog = st.progress(0, text="Starting…")
        status_box = st.empty()

    import threading

    result_holder: dict = {}

    def _call():
        try:
            resp = indexing_api.validate_dataset(dataset_id)
            result_holder["resp"] = resp
        except Exception as exc:
            result_holder["exc"] = exc
        finally:
            result_holder.setdefault("done", True)

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()

    stages = [
        (5,  "Checking registry & capabilities…"),
        (20, "Connecting to ir_datasets…"),
        (40, "Downloading / verifying files…"),
        (60, "Counting documents…"),
        (80, "Counting queries & qrels…"),
        (92, "Writing metadata.json…"),
    ]
    stage_idx = 0
    while thread.is_alive():
        if stage_idx < len(stages):
            pct, msg = stages[stage_idx]
            prog.progress(pct, text=msg)
            status_box.caption(f"⏳ {msg}")
            stage_idx += 1
        time.sleep(2)

    thread.join()
    prog.progress(100, text="Done!")
    time.sleep(0.3)
    placeholder.empty()

    st.session_state[SS_VALIDATION_IN_PROGRESS] = False

    if "exc" in result_holder:
        err = str(result_holder["exc"])
        st.error(f"❌ Validation failed: {err}")
        st.session_state[SS_DATASET_VALIDATION_RESULT] = {"error": err}
        return

    resp = result_holder.get("resp")
    if resp is None:
        st.error("❌ Validation failed: no response from indexing service.")
        st.session_state[SS_DATASET_VALIDATION_RESULT] = {"error": "no response"}
        return

    if resp.ok and isinstance(resp.data, dict):
        st.session_state[SS_DATASET_VALIDATION_RESULT] = resp.data
        gateway.clear_dataset_info_cache()
        st.success(f"✅ Dataset **{dataset_id}** validated successfully!")
        with st.expander("Validation response", expanded=False):
            st.json(resp.data)
    else:
        err = resp.error if resp else "No response"
        st.session_state[SS_DATASET_VALIDATION_RESULT] = {"error": err}
        st.error(f"❌ Validation failed: {err}")


# ---------------------------------------------------------------------------
# Section: local datasets list
# ---------------------------------------------------------------------------
def _render_local_datasets():
    st.markdown("### Local Datasets")
    st.caption("Datasets already downloaded on the server.")

    # CSS to colour the delete buttons red
    st.markdown("""
        <style>
        [data-testid="stButton"] button[kind="secondary"].delete-btn,
        div[data-delete-btn] button {
            background-color: #dc2626 !important;
            color: white !important;
            border-color: #dc2626 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh list", key="btn_refresh_local"):
        st.cache_data.clear()

    resp = indexing_api.list_local_datasets()
    if not resp.ok:
        st.warning(f"Could not fetch local datasets: {resp.error}")
        return

    datasets: list = resp.data.get("datasets", []) if isinstance(resp.data, dict) else []

    if not datasets:
        st.info("No datasets found locally.")
        return

    hdr = st.columns([4, 1, 1, 2, 1])
    for col, lbl in zip(hdr, ["Dataset ID", "Size", "Files", "Path", "Delete"]):
        col.markdown(f"**{lbl}**")

    # Single CSS block for all delete buttons — targets every button whose key starts with "del_"
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: #dc2626;
            color: white;
            border: 1px solid #dc2626;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: #b91c1c;
            border-color: #b91c1c;
        }
        </style>
    """, unsafe_allow_html=True)

    for ds in datasets:
        ds_id   = ds.get("dataset_id", "—")
        size_mb = ds.get("size_mb", 0)
        files   = ds.get("file_count", "?")
        path    = ds.get("path", "—")

        cols = st.columns([4, 1, 1, 2, 1])
        cols[0].write(ds_id)
        cols[1].write(f"{size_mb} MB")
        cols[2].write(str(files))
        cols[3].write(path)

        confirm_key = f"confirm_{ds_id}"

        with cols[4]:
            if st.button("🗑️", key=f"del_{ds_id}", help=f"Delete {ds_id}",
                         use_container_width=True):
                st.session_state[confirm_key] = True

        if st.session_state.get(confirm_key):
            st.warning(
                f"⚠️ Are you sure you want to **permanently delete** `{ds_id}`? "
                "This cannot be undone."
            )
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("Yes, delete it", type="primary", key=f"yes_{ds_id}"):
                    del_resp = indexing_api.delete_local_dataset(ds_id)
                    if del_resp.ok:
                        st.success(f"Deleted `{ds_id}`.")
                    else:
                        st.error(f"Delete failed: {del_resp.error}")
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            with no_col:
                if st.button("Cancel", key=f"no_{ds_id}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Section: upload to PostgreSQL
# ---------------------------------------------------------------------------
def _render_db_upload():
    st.markdown("### Database Management")

    # Show current database status
    db_resp = indexing_api.get_db_info()
    
    if db_resp.ok and isinstance(db_resp.data, dict):
        total_tables = db_resp.data.get("total_tables", 0)
        total_docs = db_resp.data.get("total_documents", 0)
        datasets = db_resp.data.get("datasets", [])
        
        # Database summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tables", total_tables)
        col2.metric("Total Documents", f"{total_docs:,}")
        col3.metric("Status", db_resp.data.get("status", "unknown").title())
        
        # Show datasets in database
        if datasets:
            st.markdown("#### Datasets in Database")
            
            # CSS for delete buttons
            st.markdown("""
                <style>
                div[data-testid="stButton"] > button.db-delete-btn {
                    background-color: #dc2626 !important;
                    color: white !important;
                    border: 1px solid #dc2626 !important;
                }
                div[data-testid="stButton"] > button.db-delete-btn:hover {
                    background-color: #b91c1c !important;
                    border-color: #b91c1c !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Header row
            hdr = st.columns([3, 2, 1, 1])
            for col, lbl in zip(hdr, ["Dataset ID", "Table", "Documents", "Actions"]):
                col.markdown(f"**{lbl}**")
            
            # Dataset rows
            for ds in datasets:
                ds_id = ds.get("dataset_id", "—")
                table_name = ds.get("table_name", "—")
                doc_count = ds.get("doc_count", 0)
                size = ds.get("size", "—")
                
                cols = st.columns([3, 2, 1, 1])
                cols[0].write(ds_id)
                cols[1].write(table_name)
                cols[2].write(f"{doc_count:,}")
                
                confirm_key = f"confirm_db_{ds_id}"
                
                with cols[3]:
                    if st.button("🗑️", key=f"del_db_{ds_id}", help=f"Delete {ds_id} from DB",
                                 use_container_width=True):
                        st.session_state[confirm_key] = True
                
                if st.session_state.get(confirm_key):
                    st.warning(
                        f"⚠️ Are you sure you want to **permanently delete** `{ds_id}` from database? "
                        f"This will drop the table with **{doc_count:,}** documents and cannot be undone."
                    )
                    yes_col, no_col = st.columns(2)
                    with yes_col:
                        if st.button("Yes, delete from DB", type="primary", key=f"yes_db_{ds_id}"):
                            del_resp = indexing_api.delete_dataset_from_db(ds_id)
                            if del_resp.ok:
                                st.success(f"Deleted `{ds_id}` from database.")
                            else:
                                st.error(f"Delete failed: {del_resp.error}")
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    with no_col:
                        if st.button("Cancel", key=f"no_db_{ds_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
            
            st.caption(f"💾 Total size: {sum(float(ds.get('size', '0').replace(' MB','').replace(' kB','').replace('kB','').replace('MB','')) for ds in datasets if 'size' in ds):.1f} MB")
        else:
            st.info("📭 No datasets in database yet.")
    else:
        st.warning("⚠️ Could not reach database info endpoint.")
    
    st.divider()
    
    # Upload section
    st.markdown("#### Upload Dataset to Database")
    
    # Get list of local datasets
    local_resp = indexing_api.list_local_datasets()
    local_datasets = []
    if local_resp.ok and isinstance(local_resp.data, dict):
        datasets = local_resp.data.get("datasets", [])
        local_datasets = [ds.get("dataset_id") for ds in datasets if ds.get("dataset_id")]
    
    col_id, col_btn = st.columns([3, 2])
    with col_id:
        if local_datasets:
            # Dropdown with local datasets
            default_idx = 0
            current_input = st.session_state.get("ds_selected_id", config.DATASET_NAME)
            if current_input in local_datasets:
                default_idx = local_datasets.index(current_input)
            
            upload_id = st.selectbox(
                "Select dataset to upload",
                options=local_datasets,
                index=default_idx,
                key="upload_ds_id_select",
                help="Select from locally downloaded datasets"
            )
        else:
            # Fallback to text input if no local datasets
            upload_id = st.text_input(
                "Dataset ID to upload",
                value=config.DATASET_NAME,
                key="upload_ds_id_text",
                help="No local datasets found. Download a dataset first."
            )
            st.warning("⚠️ No local datasets found. Please download a dataset first using 'Download & Validate' above.")
    
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        
        # Check if dataset is already in DB
        is_in_db = False
        if db_resp.ok and isinstance(db_resp.data, dict):
            datasets = db_resp.data.get("datasets", [])
            is_in_db = any(ds.get("dataset_id") == upload_id for ds in datasets)
        
        btn_label = "🔄 Re-upload to DB" if is_in_db else "⬆️ Upload to DB"
        btn_type = "secondary" if is_in_db else "primary"
        
        # Disable button if no local datasets
        btn_disabled = not local_datasets
        
        do_upload = st.button(btn_label, type=btn_type,
                              use_container_width=True, key="btn_upload_db",
                              disabled=btn_disabled)
    
    if is_in_db:
        st.info(f"ℹ️ Dataset `{upload_id}` is already in database. Re-uploading will drop the old table and create a new one.")

    if not do_upload:
        return

    with st.status(f"Uploading `{upload_id}` to PostgreSQL…", expanded=True) as s:
        st.write("Streaming documents in batches of 1 000…")
        up_resp = indexing_api.upload_to_db(upload_id)

    if up_resp.ok and isinstance(up_resp.data, dict):
        inserted = up_resp.data.get("inserted", 0)
        skipped  = up_resp.data.get("skipped", 0)
        table = up_resp.data.get("table", "—")
        s.update(label="✅ Upload complete!", state="complete")
        st.success(f"Inserted **{inserted:,}** documents into `{table}`, skipped **{skipped:,}**.")
        st.rerun()
    else:
        s.update(label="❌ Upload failed", state="error")
        st.error(f"Error: {up_resp.error}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown("## 📊 Dataset")

    # ── Current dataset info card ─────────────────────────────────────────
    info = gateway.get_dataset_info_cached()
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        if info:
            is_pending = info.get("validation_status") == "pending"
            source_note = " (from config)" if info.get("source") == "config_fallback" else ""
            
            c1.metric("Active Dataset", info.get("dataset_name", config.DATASET_NAME) + source_note)
            c2.metric("Documents", f"{info.get('doc_count', config.DATASET_DOC_COUNT):,}")
            c3.metric("Test Queries", f"{info.get('query_count', config.DATASET_QUERY_COUNT_APPROX):,}")
            c4.metric("Qrels", f"{info.get('qrel_count', info.get('qrels_count', config.DATASET_QREL_COUNT_APPROX)):,}")
            
            if is_pending:
                st.info("ℹ️ Using default values from config. Validate the dataset below for accurate counts.")
        else:
            c1.metric("Active Dataset", config.DATASET_NAME)
            c2.metric("Documents", f"{config.DATASET_DOC_COUNT:,}")
            c3.metric("Test Queries", f"~{config.DATASET_QUERY_COUNT_APPROX:,}")
            c4.metric("Qrels", f"~{config.DATASET_QREL_COUNT_APPROX:,}")
            st.warning("⚠️ Could not load dataset info — validate a dataset below.")

    st.divider()

    # ── Browser ───────────────────────────────────────────────────────────
    chosen = _render_browser()
    if chosen:
        st.session_state["ds_selected_id"] = chosen
        st.session_state["dataset_id_input"] = chosen
        st.toast(f"Selected: {chosen}", icon="✅")

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────
    _render_download()

    st.divider()

    # ── Local datasets list ───────────────────────────────────────────────
    _render_local_datasets()

    st.divider()

    # ── Upload to DB ──────────────────────────────────────────────────────
    _render_db_upload()

    st.divider()

    # ── Metadata viewer ───────────────────────────────────────────────────
    st.markdown("### Metadata")
    if st.button("🔄 Refresh metadata", key="btn_refresh_meta"):
        gateway.clear_dataset_info_cache()
        st.rerun()

    fresh = gateway.get_dataset_info()
    if fresh.ok and isinstance(fresh.data, dict):
        with st.expander("data/metadata.json", expanded=True):
            st.json(fresh.data)
    else:
        st.info("Metadata not available — validate a dataset first.")
