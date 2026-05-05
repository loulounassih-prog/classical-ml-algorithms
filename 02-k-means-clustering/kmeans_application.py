import numpy as np
import os
import random

from .kmeans import Point,Cloud

_BEST_RESULT = None

def find_excerpt_path() -> str|None :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "csv", "samples_excerpt.csv"),
        os.path.join(base_dir, "..", "csv", "samples_excerpt.csv"),
        "csv/samples_excerpt.csv",
        "../csv/samples_excerpt.csv",
    ]

    for path in candidates :
        if os.path.exists(path) :
            return path
    
    return None

def _load_excerpt() -> np.ndarray :
    path = find_excerpt_path()
    if path is None :
        raise FileNotFoundError("samples_excerpt.csv not found")
    
    data = np.genfromtxt(path, delimiter = ",")
    if data.ndim == 1 :
        data = data.reshape(1,-1)

    return np.array(data, dtype=float)

def _build_cloud(data: np.ndarray, k: int) -> Cloud :
    n, d = data.shape
    cloud = Cloud(n,k)
    for i in range(n) :
        p = Point(d)
        p.update_coords(data[i])
        cloud.add_point(p,0)
    
    return cloud

def _labels_array(cloud: Cloud) -> np.ndarray :
    return np.array([p.label for p in cloud.points], dtype = float)

def _centers_array(cloud: Cloud) -> np.ndarray :
    return np.array([c.coords.copy() for c in cloud.centers], dtype=float)

def _pairwise_distances(data: np.ndarray) -> np.ndarray :
    sq_norms = np.sum(data*data, axis=1, keepdim = True)
    dist_sq = sq_norms + sq_norms.T - 2*(data @ data.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)

def _silhouette_score(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return -1.0

    n = len(labels)
    clusters = {lab: np.where(labels == lab)[0] for lab in unique_labels}
    s = np.zeros(n, dtype=float)

    for i in range(n):
        my_label = labels[i]
        same_cluster = clusters[my_label]

        if len(same_cluster) <= 1:
            s[i] = 0.0
            continue

        a_i = np.sum(dist_matrix[i, same_cluster]) / (len(same_cluster) - 1)

        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == my_label:
                continue
            other_cluster = clusters[other_label]
            mean_dist = np.mean(dist_matrix[i, other_cluster])
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        if denom == 0.0:
            s[i] = 0.0
        else:
            s[i] = (b_i - a_i) / denom

    return float(np.mean(s))


def _run_one_configuration(
    data: np.ndarray,
    dist_matrix: np.ndarray,
    k: int,
    init_name: str,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    cloud = _build_cloud(data, k)

    if init_name == "random_partition":
        cloud.init_random_partition()
    elif init_name == "forgy":
        cloud.init_forgy()
    elif init_name == "plusplus":
        cloud.init_plusplus()
    else:
        raise ValueError(f"Unknown initialization: {init_name}")

    cloud.lloyd()

    labels = _labels_array(cloud)
    centers = _centers_array(cloud)
    silhouette = _silhouette_score(dist_matrix, labels)
    variance = cloud.intracluster_variance()

    return {
        "labels": labels,
        "centers": centers,
        "silhouette": silhouette,
        "variance": variance,
    }


def _search_best_parameters() -> dict:
    data = _load_excerpt()
    n, d = data.shape
    dist_matrix = _pairwise_distances(data)

    k_min = 2
    k_max = min(10, n - 1)

    init_methods = ["random_partition", "forgy", "plusplus"]
    n_runs = 5

    best_per_k = []

    for k in range(k_min, k_max + 1):
        best_init_summary = None

        for init_name in init_methods:
            runs = []

            for run_id in range(n_runs):
                seed = 1000 * k + 100 * run_id + (
                    0 if init_name == "random_partition"
                    else 1 if init_name == "forgy"
                    else 2
                )

                run = _run_one_configuration(
                    data=data,
                    dist_matrix=dist_matrix,
                    k=k,
                    init_name=init_name,
                    seed=seed,
                )
                runs.append(run)

            silhouettes = np.array([r["silhouette"] for r in runs], dtype=float)
            variances = np.array([r["variance"] for r in runs], dtype=float)

            avg_silhouette = float(np.mean(silhouettes))
            std_silhouette = float(np.std(silhouettes))
            avg_variance = float(np.mean(variances))

            best_run = max(
                runs,
                key=lambda r: (r["silhouette"], -r["variance"])
            )

            summary = {
                "k": k,
                "init": init_name,
                "avg_silhouette": avg_silhouette,
                "std_silhouette": std_silhouette,
                "avg_variance": avg_variance,
                "best_run": best_run,
            }


            if best_init_summary is None or (
                (summary["avg_silhouette"], -summary["std_silhouette"], -summary["avg_variance"])
                >
                (best_init_summary["avg_silhouette"], -best_init_summary["std_silhouette"], -best_init_summary["avg_variance"])
            ):
                best_init_summary = summary

        best_per_k.append(best_init_summary)

    best_overall = max(
        best_per_k,
        key=lambda s: (s["avg_silhouette"], -s["std_silhouette"], -s["avg_variance"])
    )

    return {
        "d_excerpt": d,
        "nb_clusters": best_overall["k"],
        "centers": best_overall["best_run"]["centers"],
    }


def _fallback_parameters(d: int, nb_samples: int):
    nb_clusters = 3 if nb_samples >= 3 else max(1, nb_samples)
    my_labels = np.zeros(nb_samples, dtype=int)
    my_center_coords = [np.zeros(d, dtype=float) for _ in range(nb_clusters)]
    return nb_clusters, my_labels, my_center_coords
  



def my_parametrization(d: int, nb_samples: int) -> (int, np.ndarray, np.ndarray):
    """Set empirically determined initial parameters for k-means."""
    global _BEST_RESULT

    if _BEST_RESULT is None:
        try:
            _BEST_RESULT = _search_best_parameters()
        except Exception:
            return _fallback_parameters(d, nb_samples)

    nb_clusters = _BEST_RESULT["nb_clusters"]
    centers = _BEST_RESULT["centers"]

    if centers.shape[1] != d:
        return _fallback_parameters(d, nb_samples)

    my_labels = np.zeros(nb_samples, dtype=int)
    my_center_coords = [centers[i].copy() for i in range(nb_clusters)]

    return (nb_clusters, my_labels, my_center_coords)