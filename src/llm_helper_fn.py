import torch
import torch.nn.functional as F
import numpy as np


# START HELPER functions ------------------------------------------------

def create_new_split(val_probs, val_labels, test_probs, test_labels, seed):
    """
    Creates a new validation/test split by randomly shuffling the union of the default splits.
    
    Args:
        shapes: [num_models, num_samples, num_classes]
    Outputs:
        shapes: [num_models, num_samples, num_classes]
    """
    # Combine along the sample dimension (assumed to be 0)
    combined_probs = torch.cat([val_probs, test_probs], dim=1)
    combined_labels = torch.cat([val_labels, test_labels], dim=0)
    total = combined_labels.shape[0]
    val_count = val_labels.shape[0]
    
    # Create a permutation using numpy's random generator with the given seed.
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(total)
    
    # Compute the new split sizes
    new_val_indices = permuted_indices[:val_count]
    new_test_indices = permuted_indices[val_count:]
    
    new_val_member_probs = combined_probs[:,new_val_indices,:]
    new_val_labels = combined_labels[new_val_indices]
    new_test_member_probs = combined_probs[:,new_test_indices,:]
    new_test_labels = combined_labels[new_test_indices]
    
    return new_val_member_probs, new_val_labels, new_test_member_probs, new_test_labels

def retrieve_data(metadataset, dataset_name, splits):
    """
       hp_candidates, indices, predictions, targets 
    """
    results = {}
    for split in splits:
        metadataset.set_state(dataset_name=dataset_name,
                        split=split)
        hp_candidates, indices = metadataset._get_hp_candidates_and_indices()
        predictions = metadataset.get_predictions(indices)
        targets = metadataset.get_targets()
        results[split] = (hp_candidates, indices, predictions, targets)
    return results





# END HELPER functions --------------------------------------------------
