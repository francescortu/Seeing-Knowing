import numpy as np
Array = np.ndarray
_N_JOBS = 16

# from src.error import MetricComputationError, DataRetrievalError
import logging
from dadapy.data import Data
import tqdm
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool
from typing import Tuple, List, Dict
import torch

class DataRetrievalError(Exception):
    """Exception raised when an issue occurs while retrieving data."""
    def __init__(self, message="Requested data not found in the file system"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
    
    
class MetricComputationError(Exception):
    """Exception raised when an issue occurs while computing the metric."""
    def __init__(self, message="An error occurred while computing the metric"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class PointOverlap():
    def __init__(self):
        pass
        
    def main(self,
             k: Int,
             input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers"]:
        """
        Compute overlap between two sets of representations.

        Returns:
            pd.DataFrame
                Dataframe containing results
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing point overlap")


        try:
            overlap = self.parallel_compute(input_i=input_i,
                                            input_j=input_j,
                                            k=k,
                                            number_of_layers=number_of_layers,
                                            parallel=parallel)
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing clustering: {e}"
            )
            raise e

        return overlap

    

    def parallel_compute(
            self,
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j:  Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            number_of_layers: Int, 
            k: Int,
            parallel: Bool = True
        ) -> Float[Array, "num_layers"]:
        """
        Compute the overlap between two set of representations for each layer.

        Inputs:
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Float[Array, "num_layers"]
        """
        assert (
            len(input_i) == len(input_j)
        ), "The two runs must have the same number of layers"
        process_layer = partial(self.process_layer,
                                input_i=input_i,
                                input_j=input_j,
                                k=k)

        if parallel:
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer)
                    for layer in tqdm.tqdm(
                        range(number_of_layers), desc="Processing layers"
                    )
                )
        else:
            results = []
            for layer in range(number_of_layers):
                results.append(process_layer(layer))

        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(
            self,
            layer: Int,predict_neighborhood_overlap_on_cache,
            # input_i: Float[Array, "num_layers num_instances d_model"] |
            #     Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            # input_j:  Float[Array, "num_layers num_instances d_model"] |
            #     Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_i,
            input_j,
            k: Int
        ) -> Float[Array, "num_layers"]:
        """
        Process a single layer
        Inputs:
            layer: Int
            input_i: Float[Array, "num_layers, num_instances, model_dim"]
            input_j: Float[Array, "num_layers, num_instances, model_dim"]
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Array
        """

        input_i = input_i[layer]
        input_j = input_j[layer]
        if isinstance(input_i, tuple):
            mat_dist_i, mat_coord_i = input_i
            data = Data(distances=(mat_dist_i, mat_coord_i), maxk=k)
            mat_dist_j, mat_coord_j = input_j
            overlap = data.return_data_overlap(distances=(mat_dist_j,
                                                          mat_coord_j), k=k)
            return overlap
        elif isinstance(input_i, np.ndarray):
            data = Data(coordinates=input_i, maxk=k)
            overlap = data.return_data_overlap(input_j, k=k)
            return overlap


class LabelOverlap():
    def __init__(self):
        pass
    def main(self,
             k: Int,
             tensors: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             labels: Int[Array, "num_layers num_instances"],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers"]:
        """
        Compute the agreement between the clustering of the hidden states
        and a given set of labels.
        Output
        ----------
        Float[Array, "num_layers"]
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing labels cluster")
        # import pdb; pdb.set_trace()
        try:
            
            output_dict = self.parallel_compute(
                number_of_layers, tensors, labels, k, parallel
            )
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing clustering: {e}"
            )
            raise e

        return output_dict

    def parallel_compute(
        self, 
        number_of_layers: Int,
        tensors: Float[Array, "num_layers num_instances d_model"] |
        Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        labels: Int[Array, "num_layers num_instances"],
        k: Int,
        parallel: Bool = True
    ) -> Float[Array, "num_layers"]:
        """
        Compute the overlap between a set of representations and a given labels
        using Advanced Peak Clustering.
        M.dErrico, E. Facco, A. Laio, A. Rodriguez, Automatic topography of
        high-dimensional data sets by non-parametric density peak clustering,
        Information Sciences 560 (2021) 476492.
        Inputs:
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            k: Int
                the number of neighbours considered for the overlap

        Returns:
            Float[Array, "num_layers"]
        """        
        process_layer = partial(
            self.process_layer, tensors=tensors, k=k,
        )
        results = []
       
        if parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer,
                                           labels=labels)
                    for layer in tqdm.tqdm(range(number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(number_of_layers)):
                results.append(process_layer(layer,
                                             labels=labels))
               
        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(
            self, 
            layer: Int,
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            labels: Float[Int, "num_instances"],
            k: Int,
    ) -> Float[Array, "num_layers"]:
        """
        Process a single layer.
        Inputs:
            layer: Int
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Float[Array, "num_layers"]
        """
        tensors = tensors[layer]
        try:
            # do clustering
            if isinstance(tensors, tuple):
                mat_dist, mat_coord = tensors
                data = Data(distances=(mat_dist, mat_coord), maxk=k)
                overlap = data.return_label_overlap(labels, k=k)
                return overlap
            elif isinstance(tensors, np.ndarray):
                # do clustering
                data = Data(coordinates=tensors, maxk=k)
                overlap = data.return_label_overlap(labels, k=k)
                return overlap
        except Exception as e:
            raise MetricComputationError(f"Error raised by Dadapy: {e}")





import torch
from typing import Dict

class PredictOverlap:
    def __init__(self, target: torch.Tensor, clusters: Dict[str, torch.Tensor], k: int = 40):
        """
        Initializes the PredictOverlap class.

        Args:
            target (torch.Tensor): Tensor of shape (num_target, 4096).
            clusters (Dict[str, torch.Tensor]): Dictionary where keys are cluster labels and values are tensors of shape (num_elem, 4096).
        """
        self.target = target  # shape (num_target, 4096)
        self.clusters = clusters

        # Prepare cluster data
        cluster_tensors = []
        cluster_labels = []
        label_to_idx = {}
        idx = 0
        for label, data in clusters.items():
            num_elems = data.size(0)
            cluster_tensors.append(data)
            cluster_labels.append(torch.full((num_elems,), idx, dtype=torch.long))
            label_to_idx[label] = idx
            idx += 1
        self.cluster_data = torch.cat(cluster_tensors, dim=0)  # shape (total_num_elem, 4096)
        self.cluster_labels = torch.cat(cluster_labels, dim=0)  # shape (total_num_elem,)
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        assert k < self.cluster_data.size(0), "k must be less than the total number of elements."
        self.k = k
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.cluster_data = self.cluster_data.to(self.device)
        self.cluster_labels = self.cluster_labels.to(self.device)
        self.target = self.target.to(self.device)

    def predict(self, x: torch.Tensor):
        """
        Computes the neighbor overlap for a single vector x.

        Args:
            x (torch.Tensor): Tensor of shape (4096,).

        Returns:
            Dict[str, int]: Dictionary with cluster labels as keys and counts as values.
        """
        x = x.to(self.device)  # shape (4096,)

        # Compute squared Euclidean distances
        distances = torch.sum((self.cluster_data - x) ** 2, dim=1)  # shape (total_num_elem,)

        # Get indices of 40 nearest neighbors
        k = 40
        nearest_indices = torch.topk(distances, k, largest=False).indices  # shape (k,)

        # Get the labels of these neighbors
        neighbor_labels = self.cluster_labels[nearest_indices]  # shape (k,)

        # Count the labels
        unique_labels, counts = neighbor_labels.unique(return_counts=True)

        # Map indices back to labels
        labels = [self.idx_to_label[idx.item()] for idx in unique_labels]
        counts = counts.tolist()

        # Return as a dict {label: count}
        overlap = dict(zip(labels, counts))
        return overlap

    def predict_avg(self):
        """
        Computes the average neighbor overlap across all target vectors.

        Returns:
            Dict[str, float]: Dictionary with cluster labels as keys and average fractions as values.
        """

        num_targets = self.target.size(0)
        batch_size = num_targets  # Adjust according to memory
        
        
        

        # Initialize dict to accumulate fractions
        cluster_fractions = {label: 0.0 for label in self.label_to_idx.keys()}

        for start in range(0, num_targets, batch_size):
            end = min(start + batch_size, num_targets)
            x_batch = self.target[start:end]  # shape (batch_size, 4096)

            # Compute distances between x_batch and cluster_data
            distances = torch.cdist(x_batch.to(torch.float32), self.cluster_data.to(torch.float32))  # shape (batch_size, total_num_elem)

            # For each vector in x_batch, get k nearest neighbors
            _, nearest_indices = torch.topk(distances, self.k, dim=1, largest=False)

            # Get neighbor labels
            neighbor_labels = self.cluster_labels[nearest_indices]  # shape (batch_size, k)

            # For each row in neighbor_labels, compute fractions
            for i in range(neighbor_labels.size(0)):
                labels, counts = neighbor_labels[i].unique(return_counts=True)
                total_counts = counts.sum().item()  # Should be equal to k
                for idx, count in zip(labels, counts):
                    label = self.idx_to_label[idx.item()]
                    fraction = count.item() / total_counts
                    cluster_fractions[label] += fraction / num_targets  # Average over all targets

        return cluster_fractions


import os
import sys
import os
import sys

# join the path of : cd .., cd .. , cd src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# join src module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from jaxtyping import Int
from typing import List
import torch

plot_config = {
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}

def average_custom_blocks(y, n):
    """
    For the plots in the main section of the paper we average the value of a certain metric for a specific layer
    over a window of n layers. This is done in order to smooth the profile.
    """
    if n==0:
        return y
    # Initialize lists to store averages
    y_avg = []

    # Handle the first block [0:n]
    
    y_avg.append(np.mean(y[0:n]))
    
    # Handle the second block [0:n+1]
    if len(y) > n:
        
        y_avg.append(np.mean(y[0:n+1]))

    # Handle subsequent blocks [i:n+i] starting from i=1
    for i in range(1, len(y)-1):
        
        y_avg.append(np.mean(y[i:n+i+1]))
    assert len(y_avg) == len(y), f"y_avg:{len(y_avg)}, y:{len(y)}"

    return np.array(y_avg)

def preprocess_label(label_array: Int[Array, "num_instances"],
                     ) -> Int[Array, "num_layers num_instances"]:
    label_array = map_label_to_int(label_array)
    
    return label_array


def map_label_to_int(my_list: List[str]
                     ) -> Int[Array, "num_layers num_instances"]:
    unique_categories = sorted(list(set(my_list)))
    category_to_int = {category: index
                       for index, category in enumerate(unique_categories)}
    numerical_list = [category_to_int[category] for category in my_list]
    numerical_array = np.array(numerical_list)
    return numerical_array



def compute_knn_cosine(X, X_new, k):
    """
    Compute the k-nearest neighbors distances and indices using cosine similarity with PyTorch.

    Parameters:
    - X (torch.Tensor): Reference tensor of shape (N, D)
    - X_new (torch.Tensor): Query tensor of shape (M, D)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances_k (torch.Tensor): Tensor of shape (M, k) containing cosine distances to the k-nearest neighbors
    - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
    """

    # Normalize the vectors to unit length
    X_norm = X / X.norm(dim=1, keepdim=True)
    X_new_norm = X_new / X_new.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = torch.mm(X_new_norm, X_norm.t())  # Shape: (M, N)

    # Convert cosine similarity to cosine distance
    cosine_dist = 1 - cosine_sim  # Shape: (M, N)

    # Find the k smallest distances and their indices
    distances_k, indices_k = torch.topk(cosine_dist, k=k, dim=1, largest=False)

    return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()

def compute_knn_euclidian(X, X_new, k):
    """
    Compute the k-nearest neighbors distances and indices using euclidian distance with PyTorch.

    Parameters:
    - X (torch.Tensor): Reference tensor of shape (N, D)
    - X_new (torch.Tensor): Query tensor of shape (M, D)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances_k (torch.Tensor): Tensor of shape (M, k) containing cosine distances to the k-nearest neighbors
    - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
    """
    X = X.float()
    X_new = X_new.float()
    
    # Compute euclidian distance
    euclidian_dist = torch.cdist(X_new, X, p=2)  # Shape: (M, N)

    # Find the k smallest distances and their indices
    distances_k, indices_k = torch.topk(euclidian_dist, k=k, dim=1, largest=False)

    return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()


# def compute_knn_euclidian(X: torch.Tensor, X_new: torch.Tensor, k: int):
#     """
#     Compute the k-nearest neighbors distances and indices using Euclidean distance with PyTorch.

#     Parameters:
#     - X (torch.Tensor): Reference tensor of shape (N, D)
#     - X_new (torch.Tensor): Query tensor of shape (M, D)
#     - k (int): Number of nearest neighbors to find

#     Returns:
#     - distances_k (torch.Tensor): Tensor of shape (M, k) containing Euclidean distances to the k-nearest neighbors
#     - indices_k (torch.Tensor): Tensor of shape (M, k) containing indices of the k-nearest neighbors
#     """
#     # Compute squared norms of X and X_new
#     x_norm_squared = (X_new ** 2).sum(dim=1).unsqueeze(1)  # Shape (M, 1)
#     y_norm_squared = (X ** 2).sum(dim=1).unsqueeze(0)      # Shape (1, N)

#     # Compute cross term
#     cross_term = torch.mm(X_new, X.t())                    # Shape (M, N)

#     # Compute pairwise squared Euclidean distances
#     distances_squared = x_norm_squared + y_norm_squared - 2 * cross_term

#     # Ensure distances are non-negative due to numerical errors
#     distances_squared = torch.clamp(distances_squared, min=0.0)

#     # Find the k nearest neighbors for each query point
#     distances_squared_k, indices_k = torch.topk(distances_squared, k=k, dim=1, largest=False)

#     # Compute the actual Euclidean distances
#     distances_k = torch.sqrt(distances_squared_k)

#     return distances_k.cpu().float().numpy(), indices_k.cpu().numpy()
