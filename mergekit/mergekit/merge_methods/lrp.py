#LRP-guided Pruning and Averaging Implementation
import torch
from typing import List, Optional
from mergekit.merge_methods.base import MergeMethod

class LRPMerge(MergeMethod):
    """
    LRP-Merge: A Pruning-Averaging hybrid that uses Layer-wise Relevance 
    Propagation (LRP) scores to identify functional importance rather 
    than simple weight magnitude.
    """

    def merge(
        self, 
        tensors:List[torch.Tensor], 
        relevances:List[torch.Tensor], 
        weights:List[float], 
        density:float = 0.01
    )->torch.Tensor:
        """
        Args:
            tensors: List of weight tensors from Global and Local models.
            relevances: Pre-calculated LRP relevance maps for each weight.
            weights: The lambda values for the weighted average (e.g., [0.7, 0.3]).
            density: The top percentage of relevant weights to preserve (default 1%).
        """
        #1. Validation
        if len(tensors)!=len(relevances):
            raise ValueError("Every weight tensor musthave a correspoding LRP relevance.")

        masked_tensors=[] #Store the pruned tensors - one for each model

        for i in range(len(tensors)):
            w=tensors[i]
            r=tensors[i]

            #2.Functional Trimming (Relevance-based Sparsification)
            #Find the threshold for the top 'density' percent of LRP scores
            k = int(density * r.numel())
            if k > 0:
                threshold = torch.kthvalue(r.flatten(),r.numel() -k+1).values
                #Creating a binary mask where relevance is below the threshold
                mask = (r >= threshold).to(w.dtype)
                #Applying mask: preserve functionally critical weights, zero out noise
                w_sparse = w * mask
            else:
                w_sparse = torch.zeros_like(w)
            
            masked_tensors.append(w_sparse)

        #3.Weighted Parameter Averaging
        #The Math:theta_new = λtheta_global + (1 - λ)theta_local
        merged_tensor = torch.zeros_like(tensors)
        total_weight = sum(weights)
        
        for i in range(len(masked_tensors)):
            normalized_lambda = weights[i] / total_weight
            merged_tensor += normalized_lambda * masked_tensors[i]

        return merged_tensor
