# file that contain tcode for computing metrix for attention heads
import torch
from typing import List, Union, Optional, Tuple, Dict
from easyroutine import Logger
from src.utils import  cosine_similarity_matrix
import statsmodels.api as sm
from collections import defaultdict
from tqdm import tqdm

def get_lower_triangular_with_diagonal(matrix):
    return [[row[i] for i in range(j + 1)] for j, row in enumerate(matrix)]


class ResidualStreamMetricComputer:
    def __init__(
        self,
        residual_stream: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        head_out_resid: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        resid_mid: bool = False,
    ):
        """
        This class implement the computation of the metrics for the activation from the residual stream of a model.
        For now, it only computes the cosine similarity between the residual stream of the different modalities.
        Args:
            residual_stream: A dictionary containing the residual stream for each input modality. 
                            The keys are the input modalities and the values are dictionaries containing the residual stream for each layer. 
                            The keys of the inner dictionary are the layer names and the values are the residual stream for that layer. An example of the structure is shown below.
            head_out_resid: A dictionary containing the output of each attention head in the residual stream (d-model). The keys are the input modalities and the values are dictionaries containing the residual stream for each layer.
            resid_mid: A boolean indicating whether the residual stream is with the middle layers or not. If True, the residual stream include also the middle layers, otherwise it is for the first and last layers.

        >>> residual_stream = {
        ...     "modality_1": {
        ...         "resid_out_0": torch.rand(10, 512),
        ...         "resid_mid_0": torch.rand(10, 512),
        ...         "resid_out_1": torch.rand(10, 512),
        ...         "resid_mid_1": torch.rand(10, 512),
        ...     },
        ...     "modality_2": {
        ...         "resid_out_0": torch.rand(10, 512),
        ...         "resid_mid_0": torch.rand(10, 512),
        ...         "resid_out_1": torch.rand(10, 512),
        ...         "resid_mid_1": torch.rand(10, 512),
        ...     },
        ... }
        >>> head_out_resid = {
            "modality_1": {
                "head_out_L0H0": torch.rand(10, 512),
                "head_out_L0H1": torch.rand(10, 512),
                ...
                "head_out_L1H0": torch.rand(10, 512),
                "head_out_L1H1": torch.rand(10, 512),
                ...
            },
            "modality_2": {
                "head_out_L0H0": torch.rand(10, 512),
                "head_out_L0H1": torch.rand(10, 512),
                ...
                "head_out_L1H0": torch.rand(10, 512),
                "head_out_L1H1": torch.rand(10, 512),
                ...
            },
        >>> ResidualStreamMetricComputer(residual_stream=residual_stream, head_out_resid=head_out_resid, resid_mid=True)

        """
        self.logger = Logger(
            logname="ResidualStreamMetricComputer",
            level="info",
            log_file_path="./logs.log",
        )
        self.resid_mid = resid_mid
        self.residual_stream = residual_stream
        self.head_out_resid = head_out_resid
        # access to one of the residual stream
        self.modalities_name = []
        self.num_layers = 0
        self.num_examples = 0

        if residual_stream:
            self.logger.info(
                f"Initializing ResidualStreamMetric with {len(residual_stream.keys())} input modalities"
            )
            self.modalities_name = list(residual_stream.keys())
            self.num_layers = (
                len([el for el in residual_stream[self.modalities_name[0]].keys() if "resid_out" in el])
                if not resid_mid
                else int(len([el for el in residual_stream[self.modalities_name[0]].keys() if "resid_out" in el]) / 2)
            )
            self.num_examples = residual_stream[self.modalities_name[0]]["resid_out_0"].shape[0]

        if head_out_resid:
            self.modalities_name = list(head_out_resid.keys())
            # NICE-TO-HAVE: Implement a more general way to extract the number of layers and heads
            self.logger.warning("The head_out_resid accept only output for full layers and heads for now, NOT arbitary heads. This should be simple to implement if needed, but now I'm in hurry.")
            self.num_layers = max([
                                    int(key.split('L')[1].split('H')[0])  # Extract 'i' and convert to int
                                    for key in head_out_resid["zh"].keys()
                                    if key.startswith('head_out_L')  # Ensure key has the correct format
                            ]) + 1
            self.num_heads = max([
                                int(key.split('H')[1])  # Extract 'i' and convert to int
                                for key in head_out_resid["zh"].keys()
                                if key.startswith('head_out_L0H')  # Ensure key has the correct format
                        ]) + 1
            
            # find the unique numebr of layers form the keys of type head_out_L0H7..
            
            self.num_examples = head_out_resid[self.modalities_name[0]]["head_out_L0H0"].shape[0]

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.logger.warning("No GPU available, using CPU")

        self._assert_streams()

    def _assert_streams(self):
        """
        Simple function to perform some checks on the residual stream and head out stream to ensure that the data is consistent
        """
        num_layer = self.num_layers 
        if self.residual_stream:
            # check that the residual stream has the same number of layers for each modality
            for modality in self.modalities_name:
                assert (
                    len([el for el in self.residual_stream[self.modalities_name[0]].keys() if "resid_out" in el]) == num_layer
                ), f"The residual stream for modality {modality} has a different number of layers than the other modalities"

            # check that the residual stream has the same number of examples for each modality
            for modality in self.modalities_name:
                assert (
                    self.residual_stream[modality][f"resid_out_0"].shape[0]
                    == self.num_examples
                ), f"The residual stream for modality {modality} has a different number of examples than the other modalities"

        if self.head_out_resid:
            # check that the head_out_stream has the same structure as the residual stream
            for modality in self.modalities_name:
                assert modality in self.head_out_resid, f"Modality {modality} is missing in head_out_stream"
                assert len(self.head_out_resid[modality]) == len(self.head_out_resid[modality]), f"Mismatch in layers between residual_stream and head_out_stream for modality {modality}"



    def _correlation_single_layer(self, stream: Dict[str, Dict[str, torch.Tensor]], resid_key: str) -> Dict[str,Tuple]:
        """
        This function computes the correlation between the residual stream for each pair of modalities for a single layer. Given the resid_key (that is the key of the layer "resid_out_0", "resid_mid_0", etc.), 
        it computes the distributions of the cosine similarity between the residual stream for each pair of modalities, i.e. the cosine similarity between all pairs of vectors in the residual stream for each modality (intra-modality) and between the residual stream of different modalities (cross-modality).
        It does not compute the correlation for the diagonal elements.
        
        Args:
            resid_key: The key of the layer for which to compute the correlation
            
        Returns:
            results: A dictionary containing the distributions of the cosine similarity for each pair of modalities. The keys are the pairs of modalities and the values are tuples containing the support and the density of the distribution.
        
        """
        # First concatenate all the modalities. We will obtain a tensor of shape (num_modalities * num_examples, model_dim)
        residual_stream = torch.cat(
            [
                stream[modality][resid_key]
                for modality in self.modalities_name
            ],
            dim=0,
        )
        
        # Compute the cosine similarity matrix for the residual stream, we will obtain a tensor of shape (num_modalities * num_examples, num_modalities * num_examples) wich contains the cosine similarity between all pairs of vectors in the residual stream
        cosine_matrix_resid = cosine_similarity_matrix(residual_stream.to(self.device))

        # Create blocks for the lower triangular part (including diagonal)
        blocks = [
            f"{modality_col} - {modality_row}"
            for i, modality_col in enumerate(self.modalities_name)
            for j, modality_row in enumerate(self.modalities_name)
            if j <= i  # This ensures we only get the lower triangular part
        ]

        results = defaultdict(tuple)
        ransk = defaultdict(tuple)
        for block in blocks:
            modality_x, modality_y = block.split(" - ")
            idx_x = self.modalities_name.index(modality_x)
            idx_y = self.modalities_name.index(modality_y)
            
            # Swap indices if necessary to ensure we're always in the lower triangle
            if idx_x < idx_y:
                idx_x, idx_y = idx_y, idx_x
            
            start_x = idx_x * self.num_examples
            end_x = (idx_x + 1) * self.num_examples
            start_y = idx_y * self.num_examples
            end_y = (idx_y + 1) * self.num_examples

            # Extract the block
            block_data = cosine_matrix_resid[start_x:end_x, start_y:end_y]
            
            # If it's not on the diagonal, we need to flatten only the lower triangular part
            mask = torch.tril(torch.ones_like(block_data), diagonal=-1).bool()
            block_data = block_data[mask]


            # Estimate the kde distribution
            # kde = sm.nonparametric.KDEUnivariate(
            #     block_data.to(torch.float32).cpu().numpy()
            # ).fit()
            # results[block] = (kde.support, kde.density)
            
            
            
            # mean of similarity
            mean = block_data.median().item()
            #std of similarity
            std = block_data.std().item()
            
            # instead of std we consider 1st and 3rd quartile
            sorted_block_data = block_data.sort()[0]
            
            q1 = sorted_block_data[int(len(sorted_block_data) * 0.25)].item()
            q3 = sorted_block_data[int(len(sorted_block_data) * 0.75)].item()
            
            results[block] = (mean, (q1, q3))

        return results
    
    def correlation_per_modality(
        self,
        analyze_heads: bool = False
    ) -> Tuple[List[Dict[str, Tuple]], Optional[List[Dict[str, Tuple]]], Optional[List[Dict[str, Tuple]]]]:
        """
        Compute the distribution of cosine_similarity between each pair of vectors in the residual stream and attention head outputs for each modality, both intra-modality and cross-modality.
        return:
            - dist_resid_out: A list of dictionaries containing the correlation for each pair of modalities for the residual stream out
                dist_resid_out[layer][block] = (support, density)
            - dist_resid_mid: A list of dictionaries containing the correlation for each pair of modalities for the residual stream mid
                dist_resid_mid[layer][block] = (support, density)
            - dist_head_out: A list of dictionaries containing the correlation for each pair of modalities for the attention head outputs
                dist_head_out[layer][head][block] = (support, density)
        """
        dist_resid_out = []
        dist_head_out = []

        # for each layer, compute correlations for residuals
        if self.residual_stream:
            for layer in tqdm(range(self.num_layers), desc="Computing correlation residual out"):
                self.logger.info(f"Computing correlation for layer {layer}")
                # estimate the distribution for that layer
                dist_single_layer = self._correlation_single_layer(
                    stream=self.residual_stream,
                    resid_key=f"resid_out_{layer}"
                )
                dist_resid_out.append(dist_single_layer)

                # if also the resid_mid (the residual stream between attn and MLP) is saved in the class, compute the correlation there
                if self.resid_mid:
                    dist_resid_mid = []
                    for layer in tqdm(range(self.num_layers), desc="Computing correlation residual mid"):
                        self.logger.info(f"Computing correlation for middle layer {layer}")
                        dist_single_layer = self._correlation_single_layer(
                            stream=self.residual_stream,
                            resid_key=f"resid_mid_{layer}"
                        )
                        dist_resid_mid.append(dist_single_layer)

        # for each layer, compute correlations for heads
        if analyze_heads and self.head_out_resid:
            for layer in tqdm(range(self.num_layers), desc="Computing correlation for heads"):
                self.logger.info(f"Computing correlation for layer {layer} heads")
                dist_head_out.append([])
                for head in range(32):
                    dist_single_head = self._correlation_single_layer(
                        stream=self.head_out_resid,
                        resid_key=f"head_out_L{layer}H{head}"
                    )
                    dist_head_out[layer].append(dist_single_head)

        return dist_resid_out, dist_resid_mid if self.resid_mid else None, dist_head_out if analyze_heads else None