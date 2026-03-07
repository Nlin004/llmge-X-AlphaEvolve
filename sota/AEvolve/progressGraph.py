import sys
import numpy as np
from typing import List, Optional

def create_sparkline_full_history(losses: List[float], width: int = 30):
    """Create sparkline showing FULL history from start to current."""
    if len(losses) < 2:
        return '▁' * width
    
    # Downsample to fit width if we have more points than width
    if len(losses) > width:
        # Take evenly spaced indices
        indices = np.linspace(0, len(losses) - 1, width, dtype=int)
        display_values = [losses[i] for i in indices]
    else:
        # Pad with empty spaces if we have fewer points
        display_values = losses.copy()
        padding = width - len(display_values)
        if padding > 0:
            # Pad with the first value on the left
            display_values = [losses[0]] * padding + display_values
    
    # Handle log scaling for better visualization of exponential decay
    vmin, vmax = min(display_values), max(display_values)
    
    if vmax == vmin:
        return '▄' * width
    
    # Use log scale if range is huge (more than 3 orders of magnitude)
    if vmin > 0 and vmax / vmin > 1000:
        display_values = np.log10(display_values)
        vmin, vmax = min(display_values), max(display_values)
    
    sparks = ' ▁▂▃▄▅▆▇█'
    sparkline = []
    
    for val in display_values:
        # Normalize and map to spark character
        normalized = (val - vmin) / (vmax - vmin)
        idx = int(normalized * 8)
        idx = max(0, min(8, idx))
        sparkline.append(sparks[idx])
    
    return ''.join(sparkline)