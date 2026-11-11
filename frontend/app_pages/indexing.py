"""
Index Management page build, monitor, and rebuild TF-IDF / BM25 / Embedding indices.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import streamlit as st

import config
import api.indexing as indexing_api
import api.gateway as gateway
from components.index_status_badge import render_index_card
from constants import SS_BUILD_IN_PROGRESS, SS_BUILD_JOB, SS_BUILD_LOG, SS_FORCE_REBUILD
from components.preprocessing_config import render_indexing_preprocessing_config
from obs import failure_hint, get_logger, log_failure
from ui_state import dataset_selector

logger = get_logger(__name__)


def _get_downloaded_datasets() -> list[str]:
    """
    Get list of downloaded datasets that are ALSO uploaded to database.
    Returns list of dataset IDs like ['beir/quora', 'cranfield', etc.]
    Only includes datasets that exist both locally and in PostgreSQL.
    """
    # Get locally downloaded datasets
    local_resp = indexing_api.list_local_datasets()
    
    if not local_resp.ok:
        return []
    
    local_datasets_info = local_resp.data.get("datasets", []) if isinstance(local_resp.data, dict) else []
    local_dataset_ids = {ds.get("dataset_id") for ds in local_datasets_info if ds.get("dataset_id")}
    
    # Get datasets in database
    db_datasets = _get_datasets_in_db()
    db_dataset_ids = set(db_datasets)
    
    # Return only datasets that are in BOTH local and database
    common_datasets = local_dataset_ids.intersection(db_dataset_ids)
    
    return sorted(common_datasets)


@st.cache_data(ttl=120, show_spinner=False)
def _get_datasets_in_db() -> list[str]:
    """Get list of datasets currently in database (cached)."""
    db_check_resp = indexing_api.get_db_info()
    if db_check_resp.ok and isinstance(db_check_resp.data, dict):
        datasets_in_db_list = db_check_resp.data.get("datasets", [])
        return [ds.get("dataset_id") for ds in datasets_in_db_list if ds.get("dataset_id")]
    return []


def _render_preprocessing_config() -> dict | None:
    """Render the preprocessing options for a build (None = server defaults)."""
    return render_indexing_preprocessing_config()


def _index_label(index_type: str) -> tuple[str, str]:
    """Human label + time estimate for an index type (``all`` included)."""
    # Special handling for 'all' index type
    if index_type == "all":
        label = "All Indices"
        estimate = "~18 min (combined)"
    else:
        label = config.RETRIEVAL_MODEL_LABELS.get(index_type, index_type)
        estimate = config.BUILD_TIME_ESTIMATES.get(index_type, "a few minutes")

    return label, estimate


# Approximate progress stages pushed to the UI while the service is working.
_BUILD_STAGES: dict[str, list[tuple[int, str]]] = {
    "all": [
        (5, "Preprocessing documents from database..."),
        (10, "Building shared preprocessing file..."),
        (20, "Preprocessing complete, starting TF-IDF..."),
        (35, "Building TF-IDF index..."),
        (45, "TF-IDF complete, starting BM25..."),
        (60, "Building BM25 inverted index..."),
        (70, "BM25 complete, starting Embedding..."),
        (75, "Loading sentence transformer model..."),
        (85, "Generating embeddings (batch 5/10)..."),
        (95, "Saving all indices..."),
        (100, "Complete!")
    ],
    "tfidf": [
        (10, "Loading preprocessed documents..."),
        (30, "Building vocabulary..."),
        (50, "Computing TF-IDF weights..."),
        (70, "Creating sparse matrix..."),
        (90, "Saving index files..."),
        (100, "Complete!")
    ],
    "bm25": [
        (10, "Loading preprocessed documents..."),
        (25, "Building inverted index..."),
        (45, "Computing document frequencies..."),
        (65, "Calculating IDF scores..."),
        (85, "Finalizing index structure..."),
        (100, "Complete!")
    ],
    "embedding": [
        (5, "Loading sentence transformer model..."),
        (10, "Model loaded, preparing documents..."),
        (30, "Generating embeddings (batch 1/10)..."),
        (50, "Generating embeddings (batch 5/10)..."),
        (70, "Generating embeddings (batch 8/10)..."),
        (90, "Saving embeddings..."),
        (100, "Complete!")
    ]
}


def _start_build_job(
    index_type: str,
    label: str,
    estimate: str,
    force: bool,
    preprocessing_config: dict | None,
    dataset_name: str | None,
    dataset_in_db: bool,
) -> dict:
    """Start the build in the background and remember it for the whole session.

    The job lives in session state (not in a local variable) so leaving the
    page no longer kills it: the thread keeps talking to the indexing service
    and the page can pick the progress back up when the user returns.
    """
    job = {
        "index_type": index_type,
        "label": label,
        "estimate": estimate,
        "dataset_in_db": dataset_in_db,
        "stages": _BUILD_STAGES.get(index_type, [(50, "Building..."), (100, "Complete!")]),
        "stage": 0,
        "started_at": time.time(),
        "done": False,
        "reported": False,
        "resp": None,
    }

    def _build_thread() -> None:
        try:
            job["resp"] = indexing_api.build_index(
                index_type,
                force_rebuild=force,
                preprocessing_config=preprocessing_config,
                dataset_name=dataset_name,
            )
        except Exception as exc:  # surfaced in the UI as a failed build
            job["resp"] = SimpleNamespace(ok=False, error=str(exc), data=None)
        finally:
            job["done"] = True

    threading.Thread(target=_build_thread, daemon=True).start()

    st.session_state[SS_BUILD_JOB] = job
    st.session_state[SS_BUILD_IN_PROGRESS] = index_type
    return job


def _elapsed_label(started_at: float) -> str:
    seconds = int(time.time() - started_at)
    return "just now" if seconds < 5 else f"{seconds}s ago"


def _render_active_build() -> None:
    """Show the build that is running now -- also after the user changed page."""
    job = st.session_state.get(SS_BUILD_JOB)
    if not job or job.get("reported"):
        return

    label = job["label"]
    log: list = st.session_state.get(SS_BUILD_LOG, [])
    stages = job["stages"]

    with st.status(f"Building {label} index (est. {job['estimate']})", expanded=True) as s:
        st.write(
            f"Calling indexing service - started {_elapsed_label(job['started_at'])}. "
            "The build keeps running if you change page."
        )

        progress_container = st.empty()
        status_container = st.empty()

        while not job["done"]:
            stage_idx = min(job["stage"], len(stages) - 1)
            pct, msg = stages[stage_idx]
            progress_container.progress(pct, text=f"{pct}% - {msg}")

            elapsed = int(time.time() - job["started_at"])
            if stage_idx > 0:
                estimated_total = elapsed / (stage_idx / len(stages))
                status_container.caption(
                    f"{msg} | Elapsed: {elapsed}s | Est. remaining: {int(estimated_total - elapsed)}s"
                )
            else:
                status_container.caption(msg)

            job["stage"] = min(stage_idx + 1, len(stages) - 1)
            time.sleep(2)

        progress_container.progress(100, text="100% - Finalizing...")
        status_container.empty()

        resp = job["resp"]
        if resp and resp.ok and isinstance(resp.data, dict):
            _report_build_success(resp, s, job["index_type"], label, log)
        else:
            _report_build_failure(
                resp, s, job["index_type"], job.get("dataset_in_db", True), label, log
            )

    job["reported"] = True
    st.session_state[SS_BUILD_IN_PROGRESS] = False
    st.session_state[SS_BUILD_LOG] = log[-50:]


def _report_build_success(resp, s, index_type: str, label: str, log: list) -> None:
    """Render a completed build (single index or the combined ``all`` run)."""
    # Handle "all" index type differently
    if index_type == "all":
        indices_results = resp.data.get("indices", {})
        success_count = resp.data.get("success_count", 0)
        total_count = resp.data.get("total_count", 3)

        if success_count == total_count:
            success_msg = f"All {total_count} indices built successfully!"
            s.update(label=success_msg, state="complete")
            st.balloons()
            log.append(f"[OK] All indices: {success_count}/{total_count} built")
        else:
            warning_msg = f"Built {success_count}/{total_count} indices"
            s.update(label=warning_msg, state="complete")
            log.append(f"[WARN] Partial build: {success_count}/{total_count}")

        # Log individual results
        for idx_name, idx_result in indices_results.items():
            if idx_result.get("status") == "success":
                log.append(f"[OK] {idx_name}: built successfully")
            elif idx_result.get("status") == "skipped":
                log.append(f"[SKIP] {idx_name}: {idx_result.get('message', 'already exists')}")
            else:
                log.append(f"[ERR] {idx_name}: {idx_result.get('message', 'unknown error')}")

        indexing_api.clear_index_status_cache()
    else:
        # Original single-index logic
        status_val = resp.data.get("status", "unknown")
        index_size = resp.data.get("index_size_mb", "N/A")
        vocab_size = resp.data.get("vocab_size", "N/A")

        if status_val == "skipped":
            s.update(
                label=f"{label} already built - use Force Rebuild to overwrite.",
                state="complete",
            )
            log.append(f"[SKIP] {label}: {resp.data.get('message')}")
        else:
            success_msg = f"{label} built successfully!"
            if index_size != "N/A":
                success_msg += f" | Size: {index_size} MB"
            if vocab_size != "N/A":
                success_msg += f" | Terms: {vocab_size:,}"

            s.update(label=success_msg, state="complete")
            st.balloons()
            log.append(f"[OK] {label}: size={index_size}MB, vocab={vocab_size}")

    indexing_api.clear_index_status_cache()


def _report_build_failure(resp, s, index_type: str, dataset_in_db: bool, label: str, log: list) -> None:
    """Render a failed build, with a dedicated panel when the dataset is missing from the DB."""
    error_msg = failure_hint(resp) if resp else "Unknown error from indexing service."
    if resp:
        log_failure(logger, f"build {label}", resp)
    s.update(label=f"{label} build failed", state="error")

    # Check if error is related to missing dataset in database AND dataset is actually not in DB
    if dataset_in_db is False and ("not uploaded to the database" in error_msg or ("dataset" in error_msg.lower() and "database" in error_msg.lower())):
        st.error("### Dataset Not Uploaded to Database")

        # Extract dataset name from error message if available
        dataset_match = error_msg.split("Dataset '")[1].split("'")[0] if "Dataset '" in error_msg else "current dataset"

        st.markdown(f"**Problem:** The dataset **`{dataset_match}`** exists locally but is not in the database.")
        st.markdown("**Impact:** Indices cannot be built because they require data from the database.")

        # Show detailed error in expander
        with st.expander("View Full Error Details", expanded=False):
            st.code(error_msg, language="text")

        with st.container(border=True):
            st.markdown("### How to Fix This Issue")
            st.markdown("""
            **Follow these steps to upload your dataset and build indices:**

            1. **Navigate** to the **Dataset Management** page (use sidebar or button below)
            2. **Locate** your dataset in the **Downloaded Datasets** section
            3. **Click** the **Upload to Database** button next to the dataset name
            4. **Monitor** the upload progress bar (may take a few minutes for large datasets)
            5. **Verify** upload completion (you'll see a success message)
            6. **Return** to this page to build the index

            **Tip:** After uploading, refresh this page to see updated index build options.
            """)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go to Dataset Management", key=f"goto_dataset_{index_type}", use_container_width=True, type="primary"):
                    st.switch_page("app_pages/dataset.py")
            with col2:
                if st.button("Refresh Status", key=f"refresh_error_{index_type}", use_container_width=True):
                    indexing_api.clear_index_status_cache()
                    st.rerun()
    else:
        # For other errors, show a clear error panel
        with st.container(border=True):
            st.markdown("### Error Details")
            st.code(error_msg, language="text")

            st.markdown("### Troubleshooting Tips")
            st.markdown("""
            - Check that the dataset is properly downloaded
            - Verify that all required services are running
            - Review the build log below for more details
            - Try refreshing the page and attempting again
            """)

    log.append(f"[ERR] {label}: {error_msg}")


def _build_single(index_type: str, force: bool, preprocessing_config: dict | None = None, dataset_in_db: bool = True, dataset_name: str = None) -> None:
    """Build one index (or all of them) with live progress and a build log entry."""
    running = st.session_state.get(SS_BUILD_JOB)
    if running and not running.get("reported"):
        st.warning(
            f"A build of **{running['label']}** is already running — showing its progress."
        )
        _render_active_build()
        return

    label, estimate = _index_label(index_type)
    _start_build_job(
        index_type,
        label,
        estimate,
        force,
        preprocessing_config,
        dataset_name,
        dataset_in_db,
    )
    _render_active_build()


def _on_dataset_change():
    """Callback when dataset selection changes."""
    selected = st.session_state.get("dataset_selector")
    previous = st.session_state.get("_previous_dataset")
    
    if selected and selected != previous:
        resp = indexing_api.validate_dataset(selected)
        if resp.ok:
            gateway.clear_dataset_info_cache()
            indexing_api.clear_index_status_cache()
            st.session_state["_previous_dataset"] = selected
            # No success message needed - dataset info will show automatically
        else:
            st.session_state["dataset_change_error"] = resp.error


def _delete_all_indices(dataset_name: str) -> None:
    """Delete all index files for a dataset using indexing service API."""
    log: list = st.session_state.get(SS_BUILD_LOG, [])
    
    with st.status("Deleting indices...", expanded=True) as s:
        if not dataset_name:
            st.warning("No dataset selected")
            s.update(label="No dataset selected", state="complete")
            return
        
        st.write(f"Deleting indices for dataset: {dataset_name}")
        
        try:
            # Call the indexing service API
            resp = indexing_api.delete_all_indices_api(dataset_name)
            
            if resp.ok and isinstance(resp.data, dict):
                status = resp.data.get("status")
                deleted_count = resp.data.get("deleted_count", 0)
                
                if status == "success" and deleted_count > 0:
                    log.append(f"[DELETE] Removed {deleted_count} index files for {dataset_name}")
                    st.write(f"✓ Deleted {deleted_count} files")
                    
                    s.update(label=f"Successfully deleted {deleted_count} index files", state="complete")
                    st.success(f"✅ All indices deleted successfully! ({deleted_count} files removed)")
                elif status == "not_found":
                    log.append(f"[DELETE] No indices found for {dataset_name}")
                    s.update(label="No indices found to delete", state="complete")
                    st.info("ℹ️ No index files were found")
                else:
                    log.append(f"[DELETE] Unexpected status: {status}")
                    s.update(label="Deletion completed with unknown status", state="complete")
                    st.info(resp.data.get("message", "Operation completed"))
            else:
                error_msg = resp.error if resp else "Unknown error from indexing service"
                log.append(f"[ERR] Delete failed: {error_msg}")
                s.update(label="Deletion failed", state="error")
                st.error(f"❌ {error_msg}")
            
            # Clear cache
            indexing_api.clear_index_status_cache()
                
        except Exception as e:
            error_msg = f"Failed to delete indices: {str(e)}"
            log.append(f"[ERR] Delete failed: {error_msg}")
            s.update(label="Deletion failed", state="error")
            st.error(f"❌ {error_msg}")
    
    st.session_state[SS_BUILD_LOG] = log[-50:]


def _delete_single_index(dataset_name: str, index_type: str) -> None:
    """Delete a specific index type for a dataset using indexing service API."""
    log: list = st.session_state.get(SS_BUILD_LOG, [])
    label = config.RETRIEVAL_MODEL_LABELS.get(index_type, index_type)
    
    with st.status(f"Deleting {label} index...", expanded=True) as s:
        if not dataset_name:
            st.warning("No dataset selected")
            s.update(label="No dataset selected", state="complete")
            return
        
        st.write(f"Deleting {label} index for dataset: {dataset_name}")
        
        try:
            # Call the indexing service API
            resp = indexing_api.delete_single_index(index_type, dataset_name)
            
            if resp.ok and isinstance(resp.data, dict):
                status = resp.data.get("status")
                deleted_count = resp.data.get("count", 0)
                deleted_files = resp.data.get("deleted_files", [])
                
                if status == "success" and deleted_count > 0:
                    log.append(f"[DELETE] Removed {label} index files: {', '.join(deleted_files)}")
                    for f in deleted_files:
                        st.write(f"  ✓ Deleted: {f}")
                    
                    s.update(label=f"Successfully deleted {label} index", state="complete")
                    st.success(f"✅ {label} index deleted successfully! ({deleted_count} files removed)")
                elif status == "not_found":
                    log.append(f"[DELETE] No {label} index files found for {dataset_name}")
                    s.update(label=f"No {label} index found", state="complete")
                    st.info(f"ℹ️ No {label} index files were found")
                else:
                    log.append(f"[DELETE] Unexpected status for {label}: {status}")
                    s.update(label="Deletion completed with unknown status", state="complete")
                    st.info(resp.data.get("message", "Operation completed"))
            else:
                error_msg = resp.error if resp else "Unknown error from indexing service"
                log.append(f"[ERR] Delete {label} failed: {error_msg}")
                s.update(label="Deletion failed", state="error")
                st.error(f"❌ {error_msg}")
            
            # Clear cache
            indexing_api.clear_index_status_cache()
                
        except Exception as e:
            error_msg = f"Failed to delete {label} index: {str(e)}"
            log.append(f"[ERR] Delete {label} failed: {error_msg}")
            s.update(label="Deletion failed", state="error")
            st.error(f"❌ {error_msg}")
    
    st.session_state[SS_BUILD_LOG] = log[-50:]


def render() -> None:
    st.markdown("## Index Management")
    _show_dataset_change_error()

    # A build started earlier keeps going while the user visits other pages;
    # this re-attaches the progress UI when they come back.
    _render_active_build()

    selected_dataset, dataset_in_db = _render_dataset_selector()
    if selected_dataset is None:
        return

    st.divider()
    preprocessing_config = _render_preprocessing_config()

    st.divider()
    force = _render_build_actions(selected_dataset, dataset_in_db, preprocessing_config)
    _render_delete_all_dialog(selected_dataset)
    _render_delete_all_execution(selected_dataset)

    st.divider()
    _render_index_status(selected_dataset, dataset_in_db, force, preprocessing_config)

    st.divider()
    _render_build_log()


def _show_dataset_change_error() -> None:
    """Report a rejected dataset change once, then drop the message."""
    # Show only error messages from dataset change
    if "dataset_change_error" in st.session_state:
        st.error(f"Validation failed: {st.session_state['dataset_change_error']}")
        del st.session_state["dataset_change_error"]


def _render_dataset_selector() -> tuple[str | None, bool]:
    """Dataset picker plus its metadata card; (None, False) when there is nothing to index."""
    # Dataset selection at the top
    available_datasets = _get_downloaded_datasets()
    
    # Get current validated dataset
    info = gateway.get_dataset_info_cached()
    current_dataset = info.get("dataset_name", config.DATASET_NAME) if info else None
    
    if not available_datasets:
        st.warning("⚠️ No datasets available for indexing")
        st.info("""
        **To build indices, you need datasets that are:**
        1. ✓ Downloaded locally (via Dataset Management page)
        2. ✓ Uploaded to PostgreSQL database
        
        **Next steps:**
        - Go to **Dataset Management** page
        - Download a dataset
        - Upload it to the database
        - Return here to build indices
        """)
        if st.button("📦 Go to Dataset Management", type="primary"):
            st.switch_page("app_pages/dataset.py")
        return None, False
    
    # Initialize previous dataset tracker
    if "_previous_dataset" not in st.session_state:
        st.session_state["_previous_dataset"] = current_dataset
    
    # Shared selector: the dataset picked on another page is preselected here
    selected_dataset = dataset_selector(
        available_datasets,
        key="dataset_selector",
        placeholder="Choose a dataset to work with...",
        on_change=_on_dataset_change,
        help="Only shows datasets that are downloaded locally AND uploaded to database",
    )
    
    # Check if dataset is uploaded to database (should always be true now, but keep for safety)
    dataset_in_db = True  # Since we filter by database presence in _get_downloaded_datasets()
    
    # Show dataset info only if a dataset is selected (use cached metadata)
    if selected_dataset:
        selected_metadata = indexing_api.get_dataset_metadata(selected_dataset)
        
        if selected_metadata:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Dataset", selected_dataset)
                
                with col2:
                    doc_count = selected_metadata.get('doc_count', info.get('doc_count', 0) if info else 0)
                    st.metric("Docs", f"{doc_count:,}")
                
                with col3:
                    query_count = selected_metadata.get('query_count', info.get('query_count', 0) if info else 0)
                    st.metric("Queries", f"{query_count:,}")
                
                with col4:
                    qrel_count = selected_metadata.get('qrel_count', selected_metadata.get('qrels_count', info.get('qrel_count', info.get('qrels_count', 0)) if info else 0))
                    st.metric("Qrels", f"{qrel_count:,}")
    else:
        st.info("Select a dataset to view its information and build indices.")
        return None, False

    return selected_dataset, dataset_in_db


def _render_build_actions(
    selected_dataset: str,
    dataset_in_db: bool,
    preprocessing_config: dict | None,
) -> bool:
    """Force checkbox, Build All (with its confirmation), Refresh Status and Delete All."""
    hdr_c1, hdr_c2, hdr_c3, hdr_c4 = st.columns([2, 2, 2, 2])
    with hdr_c1:
        force = st.checkbox(
            "Force rebuild",
            value=st.session_state.get(SS_FORCE_REBUILD, False),
            key="idx_force_rebuild",
            disabled=not dataset_in_db
        )
        st.session_state[SS_FORCE_REBUILD] = force
    with hdr_c2:
        # Disable button if dataset is not in database
        build_all_disabled = not dataset_in_db
        build_all_help = "Upload dataset to database first" if not dataset_in_db else "Build all three indices (TF-IDF, BM25, Embedding) with shared preprocessing"
        
        if st.button("Build All Indices", type="primary", use_container_width=True,
                     key="idx_build_all", disabled=build_all_disabled, help=build_all_help):
            
            # Check if any indices already exist
            existing_indices = []
            for itype in ("tfidf", "bm25", "embedding"):
                resp = indexing_api.index_status(itype, selected_dataset)
                if resp.ok and isinstance(resp.data, dict):
                    status = resp.data.get("status")
                    if status in ("ready", "built", "available"):
                        existing_indices.append(config.RETRIEVAL_MODEL_LABELS.get(itype, itype))
            
            # If indices exist and force rebuild is not enabled, show confirmation
            if existing_indices and not force:
                st.warning(f"⚠️ The following indices already exist: **{', '.join(existing_indices)}**")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("✅ Rebuild All", type="primary", key="confirm_rebuild_all", use_container_width=True):
                        st.session_state["_confirmed_rebuild_all"] = True
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Build Missing Only", key="skip_existing", use_container_width=True):
                        # Build only missing indices individually
                        for itype in ("tfidf", "bm25", "embedding"):
                            resp = indexing_api.index_status(itype, selected_dataset)
                            if resp.ok and isinstance(resp.data, dict):
                                status = resp.data.get("status")
                                if status not in ("ready", "built", "available"):
                                    _build_single(itype, False, preprocessing_config, dataset_in_db, selected_dataset)
                        st.rerun()
                
                with col3:
                    if st.button("❌ Cancel", key="cancel_rebuild", use_container_width=True):
                        st.info("Build cancelled")
            else:
                # No existing indices or force rebuild enabled - proceed directly
                st.session_state["_confirmed_rebuild_all"] = True
                st.rerun()
        
        # Execute build if confirmed
        if st.session_state.get("_confirmed_rebuild_all", False):
            st.session_state["_confirmed_rebuild_all"] = False
            _build_single("all", force, preprocessing_config, dataset_in_db, selected_dataset)
            st.rerun()
    with hdr_c3:
        if st.button("Refresh Status", use_container_width=True, key="idx_refresh"):
            indexing_api.clear_index_status_cache()
            _get_datasets_in_db.clear()
            indexing_api.clear_datasets_metadata_cache()
            st.rerun()
    
    with hdr_c4:
        if st.button("🗑️ Delete All Indices", use_container_width=True, 
                     key="idx_delete_all", type="secondary"):
            st.session_state["_show_delete_confirmation"] = True
            st.rerun()
    return force


def _render_delete_all_dialog(selected_dataset: str) -> None:
    """Confirmation dialog listing which index files would be deleted."""
    if st.session_state.get("_show_delete_confirmation", False):
        st.divider()
        st.error("### ⚠️ Confirm Deletion of Indices")
        
        if selected_dataset:
            st.markdown(f"**Dataset:** `{selected_dataset}`")
        
        # Check which indices exist
        existing_indices = []
        indices_info = []
        
        for itype in ("tfidf", "bm25", "embedding"):
            label = config.RETRIEVAL_MODEL_LABELS.get(itype, itype)
            resp = indexing_api.index_status(itype, selected_dataset) if selected_dataset else None
            
            if resp and resp.ok and isinstance(resp.data, dict):
                status = resp.data.get("status")
                size = resp.data.get("index_size_mb", "N/A")
                
                if status in ("ready", "built", "available"):
                    existing_indices.append(label)
                    indices_info.append(f"- ✓ **{label}** ({size} MB) - **Will be deleted**")
                else:
                    indices_info.append(f"- ✗ {label} - Not built")
            else:
                indices_info.append(f"- ✗ {label} - Not built")
        
        if existing_indices:
            st.markdown("**Indices status:**")
            for info in indices_info:
                st.markdown(info)
            
            st.warning(f"⚠️ **{len(existing_indices)} index file(s) will be permanently deleted from disk.** This action cannot be undone.")
        else:
            st.info("ℹ️ No indices found to delete. The operation will only clean up any remaining files.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Confirm Delete", type="primary", key="confirm_delete_indices", use_container_width=True):
                st.session_state["_show_delete_confirmation"] = False
                st.session_state["_confirmed_delete"] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", key="cancel_delete_indices", use_container_width=True):
                st.session_state["_show_delete_confirmation"] = False
                st.rerun()
        
        st.divider()


def _render_delete_all_execution(selected_dataset: str) -> None:
    """Runs the deletion once the dialog was confirmed."""
    # Execute deletion if confirmed
    if st.session_state.get("_confirmed_delete", False):
        st.session_state["_confirmed_delete"] = False
        _delete_all_indices(selected_dataset)
        st.rerun()


def _render_index_status(
    selected_dataset: str,
    dataset_in_db: bool,
    force: bool,
    preprocessing_config: dict | None,
) -> None:
    """One card per index, plus the per-index deletion confirmation."""
    st.markdown("### Index Status")
    st.caption("Build indices for the active dataset. Indices are stored as optimized files for fast retrieval.")

    # Show index cards with disabled actions if dataset not in DB
    cols = st.columns(3)
    for col, itype in zip(cols, ("tfidf", "bm25", "embedding")):
        with col:
            resp = indexing_api.index_status(itype, selected_dataset)
            status_data = resp.data if resp.ok and isinstance(resp.data, dict) else {}
            if not resp.ok:
                status_data = {"status": "unknown", "error": resp.error}
            
            # Pass dataset_in_db status to disable buttons if needed
            rebuild_clicked, delete_clicked = render_index_card(
                index_type=itype,
                status_data=status_data,
                on_rebuild_key=f"idx_rebuild_{itype}",
                on_delete_key=f"idx_delete_{itype}",
                disabled=not dataset_in_db
            )
            
            if rebuild_clicked and dataset_in_db:
                _build_single(itype, force, preprocessing_config, dataset_in_db, selected_dataset)
                st.rerun()
            
            if delete_clicked and dataset_in_db:
                st.session_state[f"_confirm_delete_{itype}"] = True
                st.rerun()
    
    # Show delete confirmation for individual indices
    for itype in ("tfidf", "bm25", "embedding"):
        if st.session_state.get(f"_confirm_delete_{itype}", False):
            st.divider()
            label = config.RETRIEVAL_MODEL_LABELS.get(itype, itype)
            st.warning(f"### ⚠️ Confirm Deletion: {label}")
            
            resp = indexing_api.index_status(itype, selected_dataset)
            if resp.ok and isinstance(resp.data, dict):
                size = resp.data.get("index_size_mb", "N/A")
                st.markdown(f"**Index:** {label}")
                st.markdown(f"**Size:** {size} MB")
                st.markdown(f"**Dataset:** `{selected_dataset}`")
            
            st.error("⚠️ This index will be permanently deleted from disk. This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Delete {label}", type="primary", key=f"confirm_delete_single_{itype}", use_container_width=True):
                    st.session_state[f"_confirm_delete_{itype}"] = False
                    _delete_single_index(selected_dataset, itype)
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", key=f"cancel_delete_single_{itype}", use_container_width=True):
                    st.session_state[f"_confirm_delete_{itype}"] = False
                    st.rerun()
            
            st.divider()
            break  # Only show one dialog at a time


def _render_build_log() -> None:
    """Collapsed build log written by `_build_single`."""
    log = st.session_state.get(SS_BUILD_LOG, [])
    if log:
        with st.expander(f"Build log ({len(log)} entries)", expanded=False):
            st.code("\n".join(reversed(log)), language="text")
        if st.button("Clear log", key="idx_clear_log"):
            st.session_state[SS_BUILD_LOG] = []
            st.rerun()