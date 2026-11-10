import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

import config
import api.clustering as clustering_api
import api.indexing as indexing_api


def _get_datasets_with_embeddings() -> List[str]:
    """
    Get list of datasets that have embeddings built.
    Returns list of dataset IDs like ['beir/quora', 'cranfield', etc.]
    """
    # Get all datasets in database
    db_info_resp = indexing_api.get_db_info()
    
    if not db_info_resp.ok:
        return []
    
    db_data = db_info_resp.data
    if not isinstance(db_data, dict):
        return []
    
    datasets_in_db = db_data.get("datasets", [])
    
    datasets_with_embeddings = []
    
    # Check each dataset for embedding index existence
    # Embeddings are stored in: /app/data/indices/{dataset_slug}/embedding/
    indices_path = Path("/app/data/indices")
    
    for dataset_info in datasets_in_db:
        if not isinstance(dataset_info, dict):
            continue
        
        dataset_id = dataset_info.get("dataset_id")
        if not dataset_id:
            continue
        
        # Convert dataset name to slug format (beir/quora -> beir_quora)
        dataset_slug = dataset_id.replace('/', '_').replace('-', '_')
        
        # Check if embedding files exist
        embedding_dir = indices_path / dataset_slug / "embedding"
        embeddings_file = embedding_dir / "embeddings.npy"
        doc_id_map_file = embedding_dir / "doc_id_map.json"
        
        if embeddings_file.exists() and doc_id_map_file.exists():
            datasets_with_embeddings.append(dataset_id)
    
    return sorted(datasets_with_embeddings)


def _display_dataset_info(dataset: str):
    """Display dataset information in a nice card."""
    # Get dataset info from database
    db_info_resp = indexing_api.get_db_info()
    
    dataset_doc_count = "N/A"
    
    if db_info_resp.ok and isinstance(db_info_resp.data, dict):
        datasets_in_db = db_info_resp.data.get("datasets", [])
        for ds_info in datasets_in_db:
            if isinstance(ds_info, dict) and ds_info.get("dataset_id") == dataset:
                dataset_doc_count = ds_info.get("doc_count", "N/A")
                break
    
    # Check embedding index file for size
    # Convert dataset name to slug format (beir/quora -> beir_quora)
    dataset_slug = dataset.replace('/', '_').replace('-', '_')
    
    indices_path = Path("/app/data/indices")
    embedding_dir = indices_path / dataset_slug / "embedding"
    embeddings_file = embedding_dir / "embeddings.npy"
    
    index_size_mb = "N/A"
    if embeddings_file.exists():
        try:
            size_bytes = embeddings_file.stat().st_size
            index_size_mb = size_bytes / (1024 * 1024)
        except:
            pass
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset", dataset)
        
        with col2:
            doc_str = f"{dataset_doc_count:,}" if isinstance(dataset_doc_count, int) else str(dataset_doc_count)
            st.metric("Documents", doc_str)
        
        with col3:
            st.metric("Embedding Dim", "384")  # Default for all-MiniLM-L6-v2
        
        with col4:
            size_str = f"{index_size_mb:.2f} MB" if isinstance(index_size_mb, (int, float)) else str(index_size_mb)
            st.metric("Index Size", size_str)


def render():
    st.title("🗂️ Document Clustering & Topic Detection")
    st.markdown(
        """
        Cluster documents based on their semantic embeddings using **HDBSCAN** and extract topics with **c-TF-IDF**.
        Visualize clusters in 2D space using UMAP dimensionality reduction.
        """
    )
    
    # Get datasets with embeddings
    available_datasets = _get_datasets_with_embeddings()
    
    if not available_datasets:
        st.warning("⚠️ No datasets with embeddings available for clustering")
        st.info("""
        **To perform clustering, you need datasets with embeddings built:**
        1. ✓ Dataset downloaded locally
        2. ✓ Dataset uploaded to database
        3. ✓ **Embedding index built**
        
        **Next steps:**
        - Go to **Indexing** page
        - Select a dataset
        - Build **Embedding Index**
        - Return here to perform clustering
        """)
        if st.button("🗂️ Go to Indexing", type="primary"):
            st.switch_page("app_pages/indexing.py")
        return
    
    # Dataset selector
    dataset = st.selectbox(
        "Select Dataset",
        options=available_datasets,
        index=0,
        key="clustering_dataset_selector",
        help="Only shows datasets that have embeddings built"
    )
    
    # Clear session state if dataset changed
    if "last_clustering_dataset" not in st.session_state:
        st.session_state["last_clustering_dataset"] = dataset
    elif st.session_state["last_clustering_dataset"] != dataset:
        # Dataset changed - clear cached results
        if "clustering_result" in st.session_state:
            del st.session_state["clustering_result"]
        st.session_state["last_clustering_dataset"] = dataset
    
    if not dataset:
        st.info("Select a dataset to perform clustering.")
        return
    
    # Display dataset info
    _display_dataset_info(dataset)
    
    st.divider()
    
    # Check clustering status
    status = _get_clustering_status(dataset)
    
    # Determine if clusters exist
    clusters_exist = status and status.get("status") == "clustered"
    
    # Show status
    st.subheader("📊 Clustering Status")
    
    if clusters_exist:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.success(f"✅ Clusters Available")
        
        with col2:
            st.info(
                f"**{status.get('n_clusters', 0)}** clusters | "
                f"**{status.get('n_noise', 0)}** noise points"
            )
        
        with col3:
            st.caption(f"Built: {status.get('timestamp', 'unknown')[:10]}")
    else:
        st.info("⚠️ No clusters built yet. Configure parameters below and click '**Run Clustering**'.")
    
    st.divider()
    
    # Clustering parameters
    st.subheader("⚙️ Clustering Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=config.CLUSTERING_MIN_CLUSTER_SIZE_RANGE[0],
            max_value=config.CLUSTERING_MIN_CLUSTER_SIZE_RANGE[1],
            value=config.CLUSTERING_MIN_CLUSTER_SIZE_DEFAULT,
            help="Minimum number of documents in a cluster",
        )
    
    with col2:
        min_samples = st.slider(
            "Min Samples",
            min_value=config.CLUSTERING_MIN_SAMPLES_RANGE[0],
            max_value=config.CLUSTERING_MIN_SAMPLES_RANGE[1],
            value=config.CLUSTERING_MIN_SAMPLES_DEFAULT,
            help="Minimum samples for core points in HDBSCAN",
        )
    
    with col3:
        metric = st.selectbox(
            "Distance Metric",
            options=config.CLUSTERING_METRICS,
            index=0,
            help="Distance metric for clustering",
        )
    
    # Show different buttons based on whether clusters exist
    if clusters_exist:

        col1, col2 = st.columns(2)
        
        with col1:
            rebuild_button = st.button(
                "🔄 Rebuild Clusters",
                type="primary",
                use_container_width=True,
                help="Rebuild clusters with new parameters (overwrites existing)"
            )
        
        with col2:
            delete_button = st.button(
                "🗑️ Delete Clusters",
                type="secondary",
                use_container_width=True,
                help="Delete existing clusters without rebuilding"
            )
        
        # Handle delete
        if delete_button:
            with st.spinner("Deleting clusters..."):
                response = clustering_api.delete_run(dataset)
                if response.ok:
                    st.success("✅ Clusters deleted successfully!")
                    if "clustering_result" in st.session_state:
                        del st.session_state["clustering_result"]
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete: {response.error}")
        
        # Handle rebuild
        run_clustering = rebuild_button
        force_recompute = True  # Always force when rebuilding
        
    else:
        # No clusters exist - show simple Run button
        run_clustering = st.button(
            "▶️ Run Clustering",
            type="primary",
            use_container_width=True,
            help="Build clusters for this dataset"
        )
        force_recompute = False
    
    # Run clustering
    if run_clustering:
        with st.spinner("Clustering documents... This may take a few minutes."):
            response = clustering_api.run(
                dataset,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                force_recompute=force_recompute,
                long_run=True,
            )
            
            if response.ok:
                st.success("✅ Clustering completed successfully!")
                st.session_state["clustering_result"] = response.data
                st.rerun()
            else:
                error_msg = response.error or "Unknown error"
                
                # Check if it's the libraries not installed error
                if "not installed" in error_msg.lower() or response.status_code == 503:
                    st.error("❌ Clustering Libraries Not Installed")
                    st.warning(
                        """
                        **The clustering feature requires additional libraries that are not yet installed.**
                        
                        To enable clustering, rebuild the Docker image:
                        ```bash
                        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build clustering
                        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d clustering
                        ```
                        
                        **Required libraries:** hdbscan, umap-learn
                        """
                    )
                else:
                    st.error(f"❌ Clustering failed: {error_msg}")
    
    st.divider()
    
    # Display results if available
    cluster_data = None
    
    # Check session state first
    if "clustering_result" in st.session_state:
        cluster_data = st.session_state["clustering_result"]
    elif status and status.get("status") == "clustered":
        # Load from backend
        cluster_data = _load_clusters(dataset)
        if cluster_data:
            st.session_state["clustering_result"] = cluster_data
    
    if cluster_data:
        _render_cluster_results(cluster_data)


def _get_clustering_status(dataset: str) -> Optional[Dict]:
    """Get clustering status for a dataset."""
    response = clustering_api.status(dataset)
    return response.data if response.ok else None


def _load_clusters(dataset: str) -> Optional[Dict]:
    """Load cluster data from backend."""
    response = clustering_api.clusters(dataset)
    return response.data if response.ok else None


def _render_cluster_results(cluster_data: Dict):
    """Render clustering results with visualization and details."""
    st.subheader("📊 Clustering Results")
    
    n_clusters = cluster_data.get("n_clusters", 0)
    n_noise = cluster_data.get("n_noise", 0)
    clusters = cluster_data.get("clusters", [])
    embeddings_2d = cluster_data.get("embeddings_2d")
    cluster_labels = cluster_data.get("cluster_labels")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", n_clusters)
    
    with col2:
        st.metric("Noise Points", n_noise)
    
    with col3:
        total_docs = len(cluster_labels) if cluster_labels else 0
        st.metric("Total Documents", total_docs)
    
    with col4:
        clustered_pct = (
            round((total_docs - n_noise) / total_docs * 100, 1)
            if total_docs > 0
            else 0
        )
        st.metric("Clustered %", f"{clustered_pct}%")
    
    st.divider()
    
    # Visualization
    if embeddings_2d and cluster_labels:
        st.subheader("🎨 2D Cluster Visualization (UMAP)")
        
        # Visualization type selector
        viz_col1, viz_col2 = st.columns([3, 1])
        with viz_col1:
            viz_type = st.selectbox(
                "Visualization Type",
                options=["Scatter Plot", "Density Contour"],
                index=0,
                help="Choose how to visualize the clusters",
            )
        
        _render_cluster_visualization(embeddings_2d, cluster_labels, clusters, viz_type)
    
    st.divider()
    
    # Cluster details
    st.subheader("📋 Cluster Details & Topics")
    
    # Sort clusters by size
    sorted_clusters = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)
    
    for cluster_info in sorted_clusters:
        cluster_id = cluster_info.get("cluster_id")
        size = cluster_info.get("size", 0)
        topic = cluster_info.get("topic")
        
        with st.expander(f"**Cluster {cluster_id}** — {size} documents"):
            if topic:
                keywords = topic.get("keywords", [])
                representative_docs = topic.get("representative_docs")
                
                st.markdown(f"**Topic Keywords:** {', '.join(keywords)}")
                
                if representative_docs:
                    st.markdown("**Representative Documents:**")
                    for i, doc in enumerate(representative_docs[:3], 1):
                        # Truncate long docs
                        doc_preview = (
                            doc[:300] + "..." if len(doc) > 300 else doc
                        )
                        st.markdown(f"{i}. {doc_preview}")
            else:
                st.info("No topic information available")


def _render_cluster_visualization(
    embeddings_2d: List[List[float]], 
    cluster_labels: List[int], 
    clusters: List[Dict],
    viz_type: str = "Scatter Plot"
):
    """Render 2D visualization of clusters with multiple display options."""
    embeddings_array = np.array(embeddings_2d)
    labels_array = np.array(cluster_labels)
    
    # Create a mapping of cluster_id to topic keywords
    cluster_topics = {}
    for cluster_info in clusters:
        cluster_id = cluster_info.get("cluster_id")
        topic = cluster_info.get("topic")
        if topic:
            keywords = topic.get("keywords", [])
            cluster_topics[cluster_id] = ", ".join(keywords[:5])
        else:
            cluster_topics[cluster_id] = f"Cluster {cluster_id}"
    
    # Add noise label
    cluster_topics[-1] = "Noise"
    
    # Create hover text
    hover_texts = [
        f"Cluster: {label}<br>Topic: {cluster_topics.get(label, 'N/A')}"
        for label in labels_array
    ]
    
    # Create color map
    unique_labels = sorted(set(labels_array))
    colors = px.colors.qualitative.Plotly
    color_map = {
        label: colors[i % len(colors)] if label != -1 else "gray"
        for i, label in enumerate(unique_labels)
    }
    
    # Render based on visualization type
    if viz_type == "Scatter Plot":
        _render_scatter_plot(embeddings_array, labels_array, unique_labels, color_map, hover_texts)
    elif viz_type == "Density Contour":
        _render_density_contour(embeddings_array, labels_array, unique_labels, color_map, hover_texts)


def _render_scatter_plot(embeddings_array, labels_array, unique_labels, color_map, hover_texts):
    """Render traditional scatter plot."""
    fig = go.Figure()
    
    for label in unique_labels:
        mask = labels_array == label
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        
        fig.add_trace(
            go.Scatter(
                x=embeddings_array[mask, 0],
                y=embeddings_array[mask, 1],
                mode="markers",
                name=label_name,
                marker=dict(
                    size=5 if label != -1 else 3,
                    color=color_map[label],
                    opacity=0.7 if label != -1 else 0.3,
                ),
                text=[
                    hover_texts[i] for i in range(len(labels_array)) if mask[i]
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    
    fig.update_layout(
        title="Document Clusters in 2D Space",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        height=600,
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each point represents a document. Colors indicate cluster assignments. "
        "Hover over points to see cluster and topic information."
    )


def _render_density_contour(embeddings_array, labels_array, unique_labels, color_map, hover_texts):
    """Render density contour plot showing cluster concentration."""
    fig = go.Figure()
    
    # Add contour for each cluster
    for label in unique_labels:
        if label == -1:  # Skip noise for contour
            continue
            
        mask = labels_array == label
        label_name = f"Cluster {label}"
        
        # Add contour layer
        fig.add_trace(
            go.Histogram2dContour(
                x=embeddings_array[mask, 0],
                y=embeddings_array[mask, 1],
                name=label_name,
                colorscale=[[0, 'rgba(255,255,255,0)'], [1, color_map[label]]],
                showscale=False,
                contours=dict(
                    coloring='fill',
                    showlabels=False,
                ),
                opacity=0.3,
            )
        )
    
    # Add scatter on top
    for label in unique_labels:
        mask = labels_array == label
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        
        fig.add_trace(
            go.Scatter(
                x=embeddings_array[mask, 0],
                y=embeddings_array[mask, 1],
                mode="markers",
                name=label_name,
                marker=dict(
                    size=4 if label != -1 else 2,
                    color=color_map[label],
                    opacity=0.6 if label != -1 else 0.2,
                    line=dict(width=0),
                ),
                text=[
                    hover_texts[i] for i in range(len(labels_array)) if mask[i]
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    
    fig.update_layout(
        title="Cluster Density Distribution",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        height=600,
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Density contours show concentration of documents. "
        "Darker/filled areas indicate higher document density within clusters."
    )
