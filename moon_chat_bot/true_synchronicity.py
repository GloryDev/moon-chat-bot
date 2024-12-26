import os
import numpy as np
from typing import List

def true_random_float(num_samples: int = 4) -> float:
    """Generate the average of n true random floats"""
    floats = []
    for _ in range(num_samples):
        random_bytes = os.urandom(8)
        random_int = int.from_bytes(random_bytes, byteorder='big')
        float_value = random_int / (2 ** 64)
        floats.append(float_value)
        # print(f"Individual float: {float_value}")  # Uncomment for debugging
    
    average = sum(floats) / num_samples
    # print(f"Average of {num_samples} samples: {average}")  # Uncomment for debugging
    return average

def true_random_multinomial(probabilities: List[float], num_samples: int = 1):
    """
    Implement multinomial sampling using true random numbers
    
    Args:
        probabilities: List of probabilities that sum to 1
        num_samples: Number of samples to draw
    
    Returns:
        List of selected indices
    """
    # Ensure probabilities sum to 1
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()
    
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(probabilities)
    
    selected_indices = []
    for _ in range(num_samples):
        # Get true random number between 0 and 1
        r = true_random_float()
        
        # Find the index where random number falls
        for idx, cum_prob in enumerate(cumulative_probs):
            if r <= cum_prob:
                selected_indices.append(idx)
                break
    
    return selected_indices