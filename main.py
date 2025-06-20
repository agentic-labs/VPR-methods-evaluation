import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize

import visualizations
import vpr_models
from test_dataset import TestDataset


def main(args):
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Perform k-means clustering if requested
    cluster_labels_db = None
    cluster_labels_queries = None
    if args.perform_clustering:
        logger.info(f"Performing k-means clustering with {args.num_clusters} clusters")
        
        # Combine all descriptors for clustering
        all_descriptors_for_clustering = np.vstack([database_descriptors, queries_descriptors])
        
        # Perform k-means clustering using FAISS
        kmeans = faiss.Kmeans(args.descriptors_dimension, args.num_clusters, niter=300, verbose=True)
        kmeans.train(all_descriptors_for_clustering.astype(np.float32))
        
        # Get cluster assignments
        _, cluster_labels = kmeans.index.search(all_descriptors_for_clustering.astype(np.float32), 1)
        cluster_labels = cluster_labels.flatten()
        
        # Split cluster labels for database and queries
        cluster_labels_db = cluster_labels[:test_ds.num_database]
        cluster_labels_queries = cluster_labels[test_ds.num_database:]
        
        # Log cluster distribution
        logger.info("Cluster distribution:")
        for i in range(args.num_clusters):
            db_count = np.sum(cluster_labels_db == i)
            query_count = np.sum(cluster_labels_queries == i)
            logger.info(f"  Cluster {i}: {db_count} database images, {query_count} query images")
        
        # Save cluster assignments
        np.save(log_dir / "cluster_labels_db.npy", cluster_labels_db)
        np.save(log_dir / "cluster_labels_queries.npy", cluster_labels_queries)
        
        # Save images organized by cluster
        cluster_dir = log_dir / "clusters"
        visualizations.save_images_by_cluster(
            test_ds.database_paths,
            test_ds.queries_paths,
            cluster_labels_db,
            cluster_labels_queries,
            args.num_clusters,
            cluster_dir
        )

    # Perform HDBSCAN clustering if requested
    hdbscan_cluster_labels_db = None
    hdbscan_cluster_labels_queries = None
    if args.perform_hdbscan:
        logger.info(f"Performing HDBSCAN clustering with min_cluster_size={args.hdbscan_min_cluster_size}, min_samples={args.hdbscan_min_samples}")
        
        try:
            import hdbscan
        except ImportError:
            logger.error("HDBSCAN not installed. Please install it with: pip install hdbscan")
            raise
        
        # Combine all descriptors for clustering
        all_descriptors_for_clustering = np.vstack([database_descriptors, queries_descriptors])
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        cluster_labels = clusterer.fit_predict(all_descriptors_for_clustering)
        
        # Split cluster labels for database and queries
        hdbscan_cluster_labels_db = cluster_labels[:test_ds.num_database]
        hdbscan_cluster_labels_queries = cluster_labels[test_ds.num_database:]
        
        # Log cluster distribution
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(cluster_labels == -1)
        
        logger.info(f"HDBSCAN found {n_clusters} clusters")
        logger.info(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        logger.info("HDBSCAN cluster distribution:")
        
        for cluster_id in unique_labels:
            db_count = np.sum(hdbscan_cluster_labels_db == cluster_id)
            query_count = np.sum(hdbscan_cluster_labels_queries == cluster_id)
            if cluster_id == -1:
                logger.info(f"  Noise: {db_count} database images, {query_count} query images")
            else:
                logger.info(f"  Cluster {cluster_id}: {db_count} database images, {query_count} query images")
        
        # Save cluster assignments
        np.save(log_dir / "hdbscan_cluster_labels_db.npy", hdbscan_cluster_labels_db)
        np.save(log_dir / "hdbscan_cluster_labels_queries.npy", hdbscan_cluster_labels_queries)
        
        # Save cluster probabilities and outlier scores
        np.save(log_dir / "hdbscan_probabilities.npy", clusterer.probabilities_)
        np.save(log_dir / "hdbscan_outlier_scores.npy", clusterer.outlier_scores_)
        
        # Save images organized by cluster
        hdbscan_cluster_dir = log_dir / "hdbscan_clusters"
        visualizations.save_hdbscan_images_by_cluster(
            test_ds.database_paths,
            test_ds.queries_paths,
            hdbscan_cluster_labels_db,
            hdbscan_cluster_labels_queries,
            hdbscan_cluster_dir
        )

    # Perform Hierarchical clustering if requested
    hierarchical_cluster_labels_db = None
    hierarchical_cluster_labels_queries = None
    linkage_matrix = None
    if args.perform_hierarchical:
        logger.info(f"Performing hierarchical clustering with average linkage and cosine distance")
        logger.info(f"Distance threshold: {args.hierarchical_distance_threshold}")
        
        # Normalize descriptors for cosine distance
        logger.info("Normalizing descriptors for cosine distance...")
        database_descriptors_norm = normalize(database_descriptors, norm='l2', axis=1)
        queries_descriptors_norm = normalize(queries_descriptors, norm='l2', axis=1)
        
        # Combine all descriptors for clustering
        all_descriptors_norm = np.vstack([database_descriptors_norm, queries_descriptors_norm])
        
        # Compute pairwise cosine distances (1 - cosine_similarity)
        logger.info("Computing pairwise cosine distances...")
        # pdist with 'cosine' metric computes 1 - cosine_similarity
        distances = pdist(all_descriptors_norm, metric='cosine')
        
        # Perform hierarchical clustering with average linkage
        logger.info("Performing hierarchical clustering...")
        linkage_matrix = linkage(distances, method='average')
        
        # Cut the dendrogram at the specified distance threshold
        if args.hierarchical_distance_threshold is not None:
            cluster_labels = fcluster(linkage_matrix, 
                                    t=args.hierarchical_distance_threshold, 
                                    criterion='distance') - 1  # Make 0-indexed
        else:
            # Use number of clusters if threshold not specified
            cluster_labels = fcluster(linkage_matrix, 
                                    t=args.hierarchical_num_clusters, 
                                    criterion='maxclust') - 1  # Make 0-indexed
        
        # Split cluster labels for database and queries
        hierarchical_cluster_labels_db = cluster_labels[:test_ds.num_database]
        hierarchical_cluster_labels_queries = cluster_labels[test_ds.num_database:]
        
        # Log cluster distribution
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        logger.info(f"Hierarchical clustering found {n_clusters} clusters")
        logger.info("Hierarchical cluster distribution:")
        
        for cluster_id in unique_labels:
            db_count = np.sum(hierarchical_cluster_labels_db == cluster_id)
            query_count = np.sum(hierarchical_cluster_labels_queries == cluster_id)
            logger.info(f"  Cluster {cluster_id}: {db_count} database images, {query_count} query images")
        
        # Save cluster assignments
        np.save(log_dir / "hierarchical_cluster_labels_db.npy", hierarchical_cluster_labels_db)
        np.save(log_dir / "hierarchical_cluster_labels_queries.npy", hierarchical_cluster_labels_queries)
        np.save(log_dir / "hierarchical_linkage_matrix.npy", linkage_matrix)
        
        # Save dendrogram visualization
        dendrogram_path = log_dir / "hierarchical_dendrogram.png"
        visualizations.plot_hierarchical_dendrogram(
            linkage_matrix, 
            dendrogram_path,
            distance_threshold=args.hierarchical_distance_threshold,
            title_suffix=" - Average Linkage with Cosine Distance"
        )
        
        # Save images organized by cluster
        hierarchical_cluster_dir = log_dir / "hierarchical_clusters"
        visualizations.save_hierarchical_images_by_cluster(
            test_ds.database_paths,
            test_ds.queries_paths,
            hierarchical_cluster_labels_db,
            hierarchical_cluster_labels_queries,
            hierarchical_cluster_dir,
            distance_threshold=args.hierarchical_distance_threshold
        )

    # Create t-SNE visualization if requested
    if args.plot_tsne:
        if args.perform_clustering and cluster_labels_db is not None:
            logger.info("Creating t-SNE visualization with k-means clustering")
            tsne_kmeans_path = log_dir / "tsne_kmeans_visualization.png"
            visualizations.plot_tsne_with_kmeans(
                database_descriptors, 
                queries_descriptors, 
                cluster_labels_db,
                cluster_labels_queries,
                tsne_kmeans_path,
                args.num_clusters
            )
        
        if args.perform_hdbscan and hdbscan_cluster_labels_db is not None:
            logger.info("Creating t-SNE visualization with HDBSCAN clustering")
            tsne_hdbscan_path = log_dir / "tsne_hdbscan_visualization.png"
            visualizations.plot_tsne_with_hdbscan(
                database_descriptors, 
                queries_descriptors, 
                hdbscan_cluster_labels_db,
                hdbscan_cluster_labels_queries,
                tsne_hdbscan_path
            )
        
        if args.perform_hierarchical and hierarchical_cluster_labels_db is not None:
            logger.info("Creating t-SNE visualization with Hierarchical clustering")
            tsne_hierarchical_path = log_dir / "tsne_hierarchical_visualization.png"
            visualizations.plot_tsne_with_hierarchical(
                database_descriptors, 
                queries_descriptors, 
                hierarchical_cluster_labels_db,
                hierarchical_cluster_labels_queries,
                tsne_hierarchical_path,
                linkage_matrix=linkage_matrix,
                distance_threshold=args.hierarchical_distance_threshold
            )
        
        # Also create standard t-SNE visualization
        logger.info("Creating standard t-SNE visualization of descriptors")
        tsne_path = log_dir / "tsne_visualization.png"
        visualizations.plot_tsne(database_descriptors, queries_descriptors, tsne_path)
        
        # Create t-SNE visualization with all nearest neighbor connections
        logger.info("Creating t-SNE visualization with nearest neighbor connections for all points")
        tsne_nn_path = log_dir / "tsne_all_nn_connections.png"
        visualizations.plot_tsne_with_all_nn_connections(
            database_descriptors, queries_descriptors, tsne_nn_path, n_neighbors=2
        )

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))
    
    # Create enhanced t-SNE visualization with connections if requested
    if args.plot_tsne:
        logger.info("Creating enhanced t-SNE visualization with query-prediction connections")
        tsne_enhanced_path = log_dir / "tsne_visualization_enhanced.png"
        visualizations.plot_tsne_with_connections(
            database_descriptors=faiss_index.reconstruct_n(0, test_ds.num_database),
            queries_descriptors=queries_descriptors,
            predictions=predictions,
            save_path=tsne_enhanced_path,
            num_connections=min(5, predictions.shape[1])  # Show top 5 connections
        )
    
    # Convert L2 distances to confidence scores
    # Lower L2 distance = higher confidence
    # We'll use exponential decay: confidence = exp(-distance)
    confidences = np.exp(-distances)
    
    # Log confidence scores for top predictions
    logger.info("Logging confidence scores for top predictions:")
    for query_idx in range(min(5, len(predictions))):  # Log first 5 queries as examples
        logger.info(f"Query {query_idx}:")
        for pred_idx in range(min(5, predictions.shape[1])):  # Top 5 predictions
            pred_id = predictions[query_idx, pred_idx]
            confidence = confidences[query_idx, pred_idx]
            distance = distances[query_idx, pred_idx]
            logger.info(f"  Prediction {pred_idx}: DB index {pred_id}, L2 distance: {distance:.4f}, Confidence: {confidence:.4f}")

    # For each query, check if the predictions are correct
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], 
            test_ds, 
            log_dir, 
            args.save_only_wrong_preds, 
            args.use_labels,
            confidences[:, : args.num_preds_to_save]  # Pass confidence scores
        )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
