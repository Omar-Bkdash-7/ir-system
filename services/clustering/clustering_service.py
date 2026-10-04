import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

# Try to import clustering libraries, disable if not available
try:
    import hdbscan
    from bertopic import BERTopic
    from umap import UMAP
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Clustering libraries (hdbscan, bertopic, umap) not available. Clustering disabled.")

from sklearn.feature_extraction.text import CountVectorizer

from services.clustering.schemas import (
    ClusterResponse,
    ClusterInfo,
    TopicInfo,
    DocumentClusterInfo,
)
from common.logging_config import setup_logging

logger = setup_logging("clustering_service")


class ClusteringService:
    def __init__(self):
        if not CLUSTERING_AVAILABLE:
            logger.error("Clustering libraries not installed. Please rebuild Docker image with updated requirements.")
        
        self.metadata_dir = Path("/app/data/clusters")
        self.embeddings_dir = Path("/app/data/indices")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ClusteringService initialized. Clusters dir: {self.metadata_dir}")

    def _load_embeddings(self, dataset: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load document embeddings from the indexing service output.
        Returns: (embeddings, doc_ids, texts)
        """
        # Convert dataset name to slug format (beir/quora -> beir_quora)
        dataset_slug = dataset.replace('/', '_').replace('-', '_')
        
        # Embeddings are stored in: data/indices/{dataset_slug}/embedding/
        embedding_dir = self.embeddings_dir / dataset_slug / "embedding"
        embeddings_path = embedding_dir / "embeddings.npy"
        doc_id_map_path = embedding_dir / "doc_id_map.json"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {embeddings_path}. "
                "Build embeddings first via indexing service."
            )
        
        if not doc_id_map_path.exists():
            raise FileNotFoundError(
                f"Document ID map not found at {doc_id_map_path}. "
                "Build embeddings first via indexing service."
            )
        
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        logger.info(f"Loading doc IDs from {doc_id_map_path}")
        with open(doc_id_map_path, "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        
        # For texts, we need to load from preprocessed documents
        # Try to load texts from database or preprocessed file
        texts = self._load_document_texts(dataset, doc_ids)
        
        if embeddings is None:
            raise ValueError(f"No embeddings found in {embeddings_path}")
        
        logger.info(
            f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
        )
        
        return embeddings, doc_ids, texts
    
    def _load_document_texts(self, dataset: str, doc_ids: List[str]) -> List[str]:
        """
        Load document texts for the given doc_ids.
        This can be from database or from preprocessed file.
        """
        # For now, use database to get texts
        try:
            from common.database import get_db_session, Document
            
            session = get_db_session()
            texts = []
            
            # Load documents in batches
            batch_size = 1000
            for i in range(0, len(doc_ids), batch_size):
                batch_doc_ids = doc_ids[i:i + batch_size]
                docs = session.query(Document).filter(
                    Document.dataset_name == dataset,
                    Document.doc_id.in_(batch_doc_ids)
                ).all()
                
                # Create a mapping
                doc_map = {doc.doc_id: doc.processed_text or doc.text for doc in docs}
                
                # Add texts in the same order as doc_ids
                for doc_id in batch_doc_ids:
                    texts.append(doc_map.get(doc_id, ""))
            
            session.close()
            logger.info(f"Loaded {len(texts)} document texts from database")
            return texts
            
        except Exception as e:
            logger.warning(f"Failed to load texts from database: {e}")
            # Fallback: use doc_ids as texts (not ideal but prevents failure)
            logger.warning("Using doc_ids as placeholder texts")
            return [f"Document {doc_id}" for doc_id in doc_ids]

    def _compute_umap_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings to 2D using UMAP for visualization.
        """
        logger.info(f"Computing UMAP 2D projection for {len(embeddings)} embeddings")
        
        umap_model = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        embeddings_2d = umap_model.fit_transform(embeddings)
        
        logger.info("UMAP projection complete")
        return embeddings_2d

    def _cluster_with_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.
        Returns cluster labels (-1 for noise).
        Supports euclidean and manhattan metrics.
        """
        logger.info(
            f"Clustering with HDBSCAN: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, metric={metric}"
        )
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        logger.info(
            f"Clustering complete: {n_clusters} clusters, {n_noise} noise points"
        )
        
        return cluster_labels

    def _extract_topics_with_bertopic(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        n_keywords: int = 10,
    ) -> Dict[int, TopicInfo]:
        """
        Extract topics from clusters using BERTopic.
        Returns mapping from cluster_id to TopicInfo.
        """
        logger.info(f"Extracting topics with BERTopic for {len(texts)} documents")
        
        # BERTopic expects cluster labels starting from 0, with -1 as outliers
        # Our HDBSCAN already provides this format
        
        # Use a simple CountVectorizer for keyword extraction
        vectorizer_model = CountVectorizer(
            stop_words="english", max_features=n_keywords * 3
        )
        
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            verbose=False,
            calculate_probabilities=False,
        )
        
        # Fit BERTopic with pre-computed embeddings and cluster labels
        # We pass y=cluster_labels to use our HDBSCAN clustering
        try:
            topics, _ = topic_model.fit_transform(texts, embeddings, y=cluster_labels)
            
            # Extract topic information
            topic_info_dict = {}
            
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    # Noise cluster
                    topic_info_dict[-1] = TopicInfo(
                        topic_id=-1,
                        keywords=["noise", "outliers"],
                        representative_docs=None,
                    )
                else:
                    # Get topic keywords for this cluster
                    topic_words = topic_model.get_topic(cluster_id)
                    
                    if topic_words:
                        keywords = [word for word, _ in topic_words[:n_keywords]]
                        
                        # Get representative documents
                        cluster_mask = cluster_labels == cluster_id
                        cluster_texts = [
                            texts[i] for i in range(len(texts)) if cluster_mask[i]
                        ]
                        
                        # Take first 3 documents as representatives (sorted by length)
                        rep_docs = sorted(cluster_texts, key=len, reverse=True)[:3]
                        
                        topic_info_dict[cluster_id] = TopicInfo(
                            topic_id=cluster_id,
                            keywords=keywords,
                            representative_docs=rep_docs,
                        )
                    else:
                        # Fallback if no topic found
                        topic_info_dict[cluster_id] = TopicInfo(
                            topic_id=cluster_id,
                            keywords=[f"cluster_{cluster_id}"],
                            representative_docs=None,
                        )
            
            logger.info(f"Extracted topics for {len(topic_info_dict)} clusters")
            return topic_info_dict
            
        except Exception as e:
            logger.warning(f"BERTopic failed: {e}. Using fallback topic extraction")
            # Fallback: simple keyword extraction per cluster
            return self._fallback_topic_extraction(
                texts, cluster_labels, n_keywords
            )

    def _fallback_topic_extraction(
        self, texts: List[str], cluster_labels: np.ndarray, n_keywords: int = 10
    ) -> Dict[int, TopicInfo]:
        """
        Fallback topic extraction using simple CountVectorizer per cluster.
        """
        logger.info("Using fallback topic extraction")
        topic_info_dict = {}
        
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if cluster_id == -1 or len(cluster_texts) == 0:
                topic_info_dict[cluster_id] = TopicInfo(
                    topic_id=cluster_id,
                    keywords=["noise"] if cluster_id == -1 else [f"cluster_{cluster_id}"],
                    representative_docs=None,
                )
            else:
                try:
                    vectorizer = CountVectorizer(
                        stop_words="english", max_features=n_keywords
                    )
                    vectorizer.fit(cluster_texts)
                    keywords = list(vectorizer.vocabulary_.keys())[:n_keywords]
                    
                    rep_docs = sorted(cluster_texts, key=len, reverse=True)[:3]
                    
                    topic_info_dict[cluster_id] = TopicInfo(
                        topic_id=cluster_id,
                        keywords=keywords,
                        representative_docs=rep_docs,
                    )
                except:
                    topic_info_dict[cluster_id] = TopicInfo(
                        topic_id=cluster_id,
                        keywords=[f"cluster_{cluster_id}"],
                        representative_docs=None,
                    )
        
        return topic_info_dict

    def _save_cluster_data(
        self,
        dataset: str,
        cluster_labels: np.ndarray,
        doc_ids: List[str],
        embeddings_2d: np.ndarray,
        topics: Dict[int, TopicInfo],
    ):
        """
        Save cluster assignments and metadata to disk.
        """
        dataset_dir = self.metadata_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster assignments
        cluster_data = {
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "cluster_labels": cluster_labels.tolist(),
            "doc_ids": doc_ids,
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "n_noise": int(np.sum(cluster_labels == -1)),
        }
        
        cluster_file = dataset_dir / "clusters.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_data, f, indent=2)
        
        # Save 2D embeddings for visualization
        embeddings_file = dataset_dir / "embeddings_2d.npy"
        np.save(embeddings_file, embeddings_2d)
        
        # Save topics (convert numpy int64 keys to regular int for JSON serialization)
        topics_data = {
            int(cluster_id): {
                "topic_id": topic.topic_id,
                "keywords": topic.keywords,
                "representative_docs": topic.representative_docs,
            }
            for cluster_id, topic in topics.items()
        }
        
        topics_file = dataset_dir / "topics.json"
        with open(topics_file, "w") as f:
            json.dump(topics_data, f, indent=2)
        
        logger.info(f"Saved cluster data to {dataset_dir}")

    def _load_cluster_data(
        self, dataset: str
    ) -> Optional[Tuple[Dict, np.ndarray, Dict[int, TopicInfo]]]:
        """
        Load cluster data from disk.
        Returns: (cluster_data, embeddings_2d, topics)
        """
        dataset_dir = self.metadata_dir / dataset
        
        cluster_file = dataset_dir / "clusters.json"
        if not cluster_file.exists():
            return None
        
        with open(cluster_file, "r") as f:
            cluster_data = json.load(f)
        
        embeddings_file = dataset_dir / "embeddings_2d.npy"
        embeddings_2d = np.load(embeddings_file) if embeddings_file.exists() else None
        
        topics_file = dataset_dir / "topics.json"
        topics_data = {}
        if topics_file.exists():
            with open(topics_file, "r") as f:
                topics_raw = json.load(f)
                topics_data = {
                    int(cluster_id): TopicInfo(**topic_info)
                    for cluster_id, topic_info in topics_raw.items()
                }
        
        return cluster_data, embeddings_2d, topics_data

    def cluster_dataset(
        self,
        dataset: str,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = "euclidean",
        force_recompute: bool = False,
    ) -> ClusterResponse:
        """
        Main method to cluster a dataset and extract topics.
        """
        if not CLUSTERING_AVAILABLE:
            raise RuntimeError(
                "Clustering libraries (hdbscan, bertopic, umap-learn) are not installed. "
                "Please rebuild the Docker image: docker-compose build clustering"
            )
        
        # Check if already clustered
        if not force_recompute:
            existing = self._load_cluster_data(dataset)
            if existing is not None:
                logger.info(f"Using cached clusters for dataset {dataset}")
                cluster_data, embeddings_2d, topics = existing
                return self._build_cluster_response(
                    cluster_data, embeddings_2d, topics
                )
        
        # Load embeddings
        embeddings, doc_ids, texts = self._load_embeddings(dataset)
        
        # Perform clustering
        cluster_labels = self._cluster_with_hdbscan(
            embeddings, min_cluster_size, min_samples, metric
        )
        
        # Extract topics
        topics = self._extract_topics_with_bertopic(texts, embeddings, cluster_labels)
        
        # Compute 2D projection for visualization
        embeddings_2d = self._compute_umap_2d(embeddings)
        
        # Save results
        self._save_cluster_data(dataset, cluster_labels, doc_ids, embeddings_2d, topics)
        
        # Build response
        cluster_data = {
            "dataset": dataset,
            "cluster_labels": cluster_labels.tolist(),
            "doc_ids": doc_ids,
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "n_noise": int(np.sum(cluster_labels == -1)),
        }
        
        return self._build_cluster_response(cluster_data, embeddings_2d, topics)

    def _build_cluster_response(
        self,
        cluster_data: Dict,
        embeddings_2d: Optional[np.ndarray],
        topics: Dict[int, TopicInfo],
    ) -> ClusterResponse:
        """
        Build ClusterResponse from loaded data.
        """
        cluster_labels = np.array(cluster_data["cluster_labels"])
        
        # Build cluster info
        clusters_info = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue  # Skip noise in summary
            
            size = int(np.sum(cluster_labels == cluster_id))
            topic = topics.get(cluster_id)
            
            clusters_info.append(
                ClusterInfo(cluster_id=cluster_id, size=size, topic=topic)
            )
        
        return ClusterResponse(
            dataset=cluster_data["dataset"],
            n_clusters=cluster_data["n_clusters"],
            n_noise=cluster_data["n_noise"],
            clusters=clusters_info,
            embeddings_2d=(
                embeddings_2d.tolist() if embeddings_2d is not None else None
            ),
            cluster_labels=cluster_labels.tolist(),
        )

    def get_clusters(self, dataset: str) -> Optional[ClusterResponse]:
        """
        Get existing cluster data for a dataset.
        """
        existing = self._load_cluster_data(dataset)
        if existing is None:
            return None
        
        cluster_data, embeddings_2d, topics = existing
        return self._build_cluster_response(cluster_data, embeddings_2d, topics)

    def get_document_cluster(
        self, dataset: str, doc_id: str
    ) -> Optional[DocumentClusterInfo]:
        """
        Get cluster and topic info for a specific document.
        """
        existing = self._load_cluster_data(dataset)
        if existing is None:
            return None
        
        cluster_data, _, topics = existing
        doc_ids = cluster_data["doc_ids"]
        cluster_labels = cluster_data["cluster_labels"]
        
        try:
            doc_index = doc_ids.index(doc_id)
            cluster_id = cluster_labels[doc_index]
            topic = topics.get(cluster_id)
            
            return DocumentClusterInfo(
                doc_id=doc_id, cluster_id=cluster_id, topic=topic
            )
        except ValueError:
            return None

    def list_clustered_datasets(self) -> List[str]:
        """
        List all datasets that have been clustered.
        """
        if not self.metadata_dir.exists():
            return []
        
        datasets = []
        for item in self.metadata_dir.iterdir():
            if item.is_dir() and (item / "clusters.json").exists():
                datasets.append(item.name)
        
        return sorted(datasets)

    def delete_clusters(self, dataset: str) -> bool:
        """
        Delete cluster data for a dataset.
        """
        dataset_dir = self.metadata_dir / dataset
        
        if not dataset_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(dataset_dir)
        logger.info(f"Deleted cluster data for dataset {dataset}")
        return True

    def get_status(self, dataset: str) -> Dict:
        """
        Get clustering status for a dataset.
        """
        existing = self._load_cluster_data(dataset)
        
        if existing is None:
            return {
                "dataset": dataset,
                "status": "not_clustered",
                "message": "Dataset has not been clustered yet",
            }
        
        cluster_data, _, _ = existing
        
        return {
            "dataset": dataset,
            "status": "clustered",
            "n_clusters": cluster_data["n_clusters"],
            "n_noise": cluster_data["n_noise"],
            "timestamp": cluster_data.get("timestamp", "unknown"),
        }
