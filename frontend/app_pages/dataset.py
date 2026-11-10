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
from ui_style import inject_style

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

    # Get list of datasets from the catalogue
    dataset_ids = [ds[0] for ds in CATALOGUE]
    
    # Get list of already downloaded datasets
    local_resp = indexing_api.list_local_datasets()
    downloaded_ids = set()
    if local_resp.ok and isinstance(local_resp.data, dict):
        datasets = local_resp.data.get("datasets", [])
        downloaded_ids = {ds.get("dataset_id") for ds in datasets if ds.get("dataset_id")}

    col_id, col_btn = st.columns([3, 2])
    with col_id:
        # Get the default selection
        default_idx = None
        selected_id = st.session_state.get("ds_selected_id")
        if selected_id and selected_id in dataset_ids:
            default_idx = dataset_ids.index(selected_id)
        
        dataset_id: str = st.selectbox(
            "Dataset ID",
            options=dataset_ids,
            index=default_idx,
            placeholder="Choose a dataset...",
            key="dataset_id_input",
            help="Select dataset from ir_datasets catalogue"
        )
        
        # Show download status if dataset is already downloaded
        if dataset_id and dataset_id in downloaded_ids:
            st.info(f"ℹ️ Dataset `{dataset_id}` is already downloaded locally.")
        elif dataset_id:
            st.caption("⬇️ Dataset will be downloaded from ir_datasets.")
    
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        do_download = st.button("⬇️ Download & Validate", type="primary",
                                use_container_width=True, key="btn_dl", 
                                disabled=not dataset_id)

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
        (92, "Saving metadata…"),
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
    inject_style("dataset_local_actions")

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
    inject_style("dataset_browser_rows")

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
                        st.success(f"✅ Deleted `{ds_id}`.")
                        gateway.clear_dataset_info_cache()
                        indexing_api.clear_index_status_cache()
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
            inject_style("dataset_db_actions")
            
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
                                st.success(f"✅ Deleted `{ds_id}` from database.")
                                gateway.clear_dataset_info_cache()
                                indexing_api.clear_index_status_cache()
                            else:
                                st.error(f"Delete failed: {del_resp.error}")
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    with no_col:
                        if st.button("Cancel", key=f"no_db_{ds_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
            
            # Calculate total size properly
            def parse_size_to_mb(size_str):
                """Convert PostgreSQL size string to MB."""
                if not size_str or size_str == '—':
                    return 0.0
                
                # Remove all spaces
                size_str = size_str.replace(' ', '')
                
                # Extract number and unit
                import re
                match = re.match(r'([\d.]+)([A-Za-z]+)', size_str)
                if not match:
                    return 0.0
                
                value = float(match.group(1))
                unit = match.group(2).upper()
                
                # Convert to MB
                if unit in ['B', 'BYTES']:
                    return value / (1024 * 1024)
                elif unit in ['KB', 'KIB']:
                    return value / 1024
                elif unit in ['MB', 'MIB']:
                    return value
                elif unit in ['GB', 'GIB']:
                    return value * 1024
                elif unit in ['TB', 'TIB']:
                    return value * 1024 * 1024
                else:
                    return 0.0
            
            total_size_mb = sum(parse_size_to_mb(ds.get('size', '0')) for ds in datasets)
            
            if total_size_mb >= 1024:
                st.caption(f"💾 Total size: {total_size_mb / 1024:.2f} GB")
            else:
                st.caption(f"💾 Total size: {total_size_mb:.1f} MB")
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
            # Dropdown with local datasets and placeholder
            upload_id = st.selectbox(
                "Select dataset to upload",
                options=local_datasets,
                index=None,
                placeholder="Choose a dataset...",
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
        # Check if dataset is already in DB
        is_in_db = False
        if upload_id and db_resp.ok and isinstance(db_resp.data, dict):
            datasets = db_resp.data.get("datasets", [])
            is_in_db = any(ds.get("dataset_id") == upload_id for ds in datasets)
        
        btn_label = "🔄 Re-upload to DB" if is_in_db else "⬆️ Upload to DB"
        btn_type = "secondary" if is_in_db else "primary"
        
        # Disable button if no dataset selected or no local datasets
        btn_disabled = not local_datasets or not upload_id
        
        do_upload = st.button(btn_label, type=btn_type,
                              use_container_width=True, key="btn_upload_db",
                              disabled=btn_disabled)
    
    if upload_id and is_in_db:
        st.info(f"ℹ️ Dataset `{upload_id}` is already in database. Re-uploading will drop the old table and create a new one.")

    if not do_upload or not upload_id:
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
        gateway.clear_dataset_info_cache()
        st.rerun()
    else:
        s.update(label="❌ Upload failed", state="error")
        error_msg = up_resp.error or "Unknown error"
        
        # Check if it's a corruption error
        if "corrupted" in error_msg.lower() or "decompression" in error_msg.lower() or "lz4" in error_msg.lower():
            st.error("### 🔴 Dataset Files Corrupted")
            
            st.markdown("""
            The dataset files appear to be **corrupted or incomplete**. This usually happens when:
            - Download was interrupted
            - Network connection was lost during download
            - Disk space was insufficient
            """)
            
            st.warning("**📋 Solution:** Click the button below to automatically fix the dataset.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔧 Fix Corrupted Dataset", type="primary", use_container_width=True, key="fix_corrupted"):
                    with st.spinner(f"Fixing corrupted dataset `{upload_id}`..."):
                        st.info("⏳ Deleting corrupted files...")
                        fix_resp = indexing_api.fix_corrupted_dataset(upload_id)
                        
                        if fix_resp.ok and isinstance(fix_resp.data, dict):
                            counts = fix_resp.data.get("counts", {})
                            docs = counts.get("documents", 0)
                            queries = counts.get("queries", 0)
                            qrels = counts.get("qrels", 0)
                            
                            st.success(f"""
                            ✅ **Dataset fixed successfully!**
                            
                            - Documents: {docs:,}
                            - Queries: {queries:,}
                            - Qrels: {qrels:,}
                            """)
                            
                            st.info("💡 You can now try uploading to database again.")
                            gateway.clear_dataset_info_cache()
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to fix dataset: {fix_resp.error}")
            
            with col2:
                with st.expander("🔍 See error details"):
                    st.code(error_msg, language="text")
        else:
            st.error(f"**Error:** {error_msg}")
            
            with st.expander("🔍 See full error"):
                st.code(error_msg, language="text")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown("## 📊 Dataset Management")

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
    st.markdown("### Metadata Files")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Refresh", key="btn_refresh_meta", use_container_width=True):
            gateway.clear_dataset_info_cache()
            st.rerun()
    
    # Get list of validated datasets for metadata viewer
    metadata_resp = indexing_api.list_datasets_metadata()
    
    if metadata_resp.ok and isinstance(metadata_resp.data, dict):
        all_datasets_meta = metadata_resp.data.get("datasets", [])
        
        if all_datasets_meta:
            # Create dropdown with all available datasets
            dataset_names = [ds.get("dataset_name") for ds in all_datasets_meta if ds.get("dataset_name")]
            
            selected_metadata_ds = st.selectbox(
                "Select dataset to view metadata",
                options=dataset_names,
                index=None,
                placeholder="Choose a dataset...",
                key="metadata_viewer_selector",
                help="Choose a dataset to view its detailed metadata"
            )
            
            # Display metadata if a dataset is selected
            if selected_metadata_ds:
                # Find the selected dataset's metadata
                selected_ds_meta = next((ds for ds in all_datasets_meta if ds.get("dataset_name") == selected_metadata_ds), None)
                
                if selected_ds_meta:
                    with st.expander(f"📄 {selected_metadata_ds}", expanded=True):
                        st.json(selected_ds_meta)
                else:
                    st.warning(f"Metadata not found for: {selected_metadata_ds}")
        else:
            st.info("No validated datasets available. Validate a dataset first to view its metadata.")
    else:
        st.warning("Could not retrieve metadata list.")

