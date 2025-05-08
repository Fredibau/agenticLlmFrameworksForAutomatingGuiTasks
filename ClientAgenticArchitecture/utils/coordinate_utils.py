"""
Utility functions for handling screen coordinates.
"""
from typing import Tuple

def constrain_to_primary_screen(x: int, y: int, primary_width: int, primary_height: int) -> Tuple[int, int]:
    """
    Convert normalized coordinates (0-1000 range) to pixel coordinates
    and ensure they stay within the primary screen boundaries.
    
    Args:
        x: X coordinate in 0-1000 range
        y: Y coordinate in 0-1000 range
        primary_width: Width of the primary screen in pixels
        primary_height: Height of the primary screen in pixels
        
    Returns:
        (x, y) tuple with pixel coordinates
    """
    # Convert from 0-1000 range to pixel coordinates
    pixel_x = int((x / 1000.0) * primary_width)
    pixel_y = int((y / 1000.0) * primary_height)
    
    # Constrain to screen boundaries
    pixel_x = max(0, min(pixel_x, primary_width - 1))
    pixel_y = max(0, min(pixel_y, primary_height - 1))
    
    return pixel_x, pixel_y