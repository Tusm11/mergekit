import os
import torch
import tempfile
import pytest

from mergekit.config import (
    InputModelDefinition,
    MergeConfiguration,
)
from tests.common import make_picollama
from mergekit.io.lazy_tensor_loader import LazyTensorLoader
from mergekit.merge import MergeOptions, run_merge

@pytest.fixture(scope="session")
def model_a(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_a"))

@pytest.fixture(scope="session")
def model_b(tmp_path_factory):
    return make_picollama(tmp_path_factory.mktemp("model_b"))

def test_lrp_merge_differs_from_magnitude(model_a, model_b):
    """
    Test that providing LRP scores produces a different mask 
    than magnitude-based importance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        lrp_path_inv = os.path.join(tmpdir, "lrp_scores_inv.pt")
        lrp_path_mag = os.path.join(tmpdir, "lrp_scores_mag.pt")
        
        loader = LazyTensorLoader.from_disk(model_b)
        
        lrp_scores_inv = {}
        lrp_scores_mag = {}
        for name in loader.index.tensor_paths:
            tensor = loader.get_tensor(name)
            # Inverse of magnitude to ensure LRP mask differs from magnitude
            lrp_scores_inv[name] = 1.0 / (torch.abs(tensor) + 1e-6)
            # Direct magnitude
            lrp_scores_mag[name] = torch.abs(tensor)
            
        torch.save(lrp_scores_inv, lrp_path_inv)
        torch.save(lrp_scores_mag, lrp_path_mag)
        
        # Run LRP merge with inverse scores
        config_lrp = MergeConfiguration(
            merge_method="lrp",
            base_model=model_a,
            models=[
                InputModelDefinition(
                    model=model_b,
                    parameters={"weight": 1.0, "lrp_scores": lrp_path_inv},
                ),
            ],
            parameters={"density": 0.5},
            dtype="bfloat16",
        )
        
        # Run LRP merge with magnitude scores (simulating the old fallback)
        config_mag = MergeConfiguration(
            merge_method="lrp",
            base_model=model_a,
            models=[
                InputModelDefinition(
                    model=model_b,
                    parameters={"weight": 1.0, "lrp_scores": lrp_path_mag},
                ),
            ],
            parameters={"density": 0.5},
            dtype="bfloat16",
        )
        
        out_dir_lrp = os.path.join(tmpdir, "out_lrp")
        out_dir_mag = os.path.join(tmpdir, "out_mag")
        
        run_merge(config_lrp, out_path=out_dir_lrp, options=MergeOptions())
        run_merge(config_mag, out_path=out_dir_mag, options=MergeOptions())
        
        loader_lrp = LazyTensorLoader.from_disk(out_dir_lrp, lazy_unpickle=False)
        loader_mag = LazyTensorLoader.from_disk(out_dir_mag, lazy_unpickle=False)
        
        diff_found = False
        for name in loader_lrp.index.tensor_paths:
            if name.endswith("lm_head.weight") or name.endswith("embed_tokens.weight"):
                continue  # some layers might not be sparseified based on architecture, ignore them or just check all
            t_lrp = loader_lrp.get_tensor(name)
            t_mag = loader_mag.get_tensor(name)
            
            if not torch.equal(t_lrp, t_mag):
                diff_found = True
                break
                
        assert diff_found, "LRP mask produced the same output as magnitude fallback!"
