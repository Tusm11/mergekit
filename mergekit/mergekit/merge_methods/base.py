# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from typing_extensions import TypeAlias

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.tokenizer import PermutedEmbeddings
from mergekit.merge_methods.base import MergeMethod,MergeTensorInput,ConfigParameterDef


class TensorDictWrapper(Task[Dict[ModelReference, torch.Tensor]]):
    tensors: ImmutableMap[ModelReference, Task[torch.Tensor]]

    def arguments(self) -> Dict[str, Task]:
        return {
            k.model_dump_json(
                exclude_none=True, exclude_defaults=True, round_trip=True
            ): v
            for k, v in self.tensors.items()
        }

    def execute(self, **kwargs) -> Dict[ModelReference, torch.Tensor]:
        return {ModelReference.model_validate_json(k): v for k, v in kwargs.items()}


MergeTensorInput: TypeAlias = Union[
    GatherTensors, PermutedEmbeddings, TensorDictWrapper
]


class ConfigParameterDef(BaseModel):
    name: str
    required: bool = False
    default_value: Any = None


class MergeMethod(ABC):
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return []

    def parameters(self) -> List[ConfigParameterDef]:
        return []

    @abstractmethod
    def name(self) -> str: ...

    def pretty_name(self) -> Optional[str]:
        return None

    def reference_url(self) -> Optional[str]:
        return None

    @abstractmethod
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmgutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task: ...

class LRPMergeTask(Task[torch.Tensor]):
    """
    The execution task for LRP-Merge. It performs functional trimming 
    using relevance scores followed by weighted parameter averaging.
    """
    def execute(self, **kwargs) -> torch.Tensor:
        # 1. Parse inputs
        # Standard mergekit tasks receive a dict of tensors keyed by ModelReference
        model_tensors = kwargs.get("tensors", {})
        # Pre-calculated LRP relevance maps (should be loaded via plan.py)
        relevance_maps = kwargs.get("relevances", {}) 
        
        # Parameters from YAML
        density = kwargs.get("density", 0.01) # Default to top 1% [3]
        weights = kwargs.get("weights", {})
        
        masked_tensors = []
        model_refs = list(model_tensors.keys())

        for ref in model_refs:
            w = model_tensors[ref]
            r = relevance_maps[ref] # Loaded functional relevance tensor

            # 2. Functional Trimming (LRP-based Sparsification) [4]
            # Find threshold for the top 'density' percent of LRP scores
            k = int(density * r.numel())
            if k > 0:
                threshold = torch.kthvalue(r.flatten(), r.numel() - k + 1).values
                # Binary mask: preserve weights that 'move the needle' on AD markers
                mask = (r >= threshold).to(w.dtype)
                w_sparse = w * mask
            else:
                w_sparse = torch.zeros_like(w)
            
            masked_tensors.append((w_sparse, weights.get(ref, 1.0)))

        # 3. Weighted Parameter Averaging [5, 6]
        # Math: θ_new = λ_global * θ_global_sparse + λ_local * θ_local_sparse
        merged_tensor = torch.zeros_like(list(model_tensors.values()))
        total_weight = sum(w for _, w in masked_tensors)
        
        for tensor, weight in masked_tensors:
            normalized_lambda = weight / total_weight
            merged_tensor += normalized_lambda * tensor

        return merged_tensor

class LRPMerge(MergeMethod):
    def name(self) -> str:
        return "lrp"

    def pretty_name(self) -> str:
        return "LRP-Guided Relevance Merging"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", default_value=0.01) # Relevance threshold [3]
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", default_value=1.0) # λ factor per model [5]
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        # Create the computational graph task [7]
        return LRPMergeTask(
            tensors=tensors,
            density=parameters.get("density"),
            weights={m: p.get("weight") for m, p in tensor_parameters.items()}
        )
