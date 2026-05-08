from typing import Any

import torch


def collate_upfall_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function to handle None pose values in UPFall batches."""
    result = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Handle None values (e.g., missing pose_cam1/pose_cam2)
        if key in ("pose_cam1", "pose_cam2"):
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                # All None, keep as is
                result[key] = None
            elif len(valid_values) == len(values):
                # All present, stack normally
                result[key] = torch.stack(valid_values)
            else:
                # Mixed None and tensors - use first valid as template, pad with zeros
                template = valid_values[0]
                padded = []
                for v in values:
                    if v is None:
                        padded.append(torch.zeros_like(template))
                    else:
                        padded.append(v)
                result[key] = torch.stack(padded)
        elif key == "id":
            # Keep as list of strings
            result[key] = values
        elif key == "label":
            # Stack labels
            result[key] = torch.stack(values)
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            result[key] = torch.stack(values)
        else:
            # Keep as list
            result[key] = values
    
    return result
