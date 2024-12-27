from typing import List, Tuple

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox: List of [x1, y1, x2, y2] coordinates
    
    Returns:
        Tuple of (center_x, center_y) coordinates
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_bbox_width(bbox: List[float]) -> float:
    """
    Get the width of a bounding box.
    
    Args:
        bbox: List of [x1, y1, x2, y2] coordinates
    
    Returns:
        Width of the bounding box
    """
    return bbox[2] - bbox[0]

def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point as (x, y) coordinates
        p2: Second point as (x, y) coordinates
    
    Returns:
        Euclidean distance between the points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate the x and y distances between two points.
    
    Args:
        p1: First point as (x, y) coordinates
        p2: Second point as (x, y) coordinates
    
    Returns:
        Tuple of (x_distance, y_distance)
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox: List[float]) -> Tuple[int, int]:
    """
    Get the foot position from a bounding box (center bottom point).
    
    Args:
        bbox: List of [x1, y1, x2, y2] coordinates
    
    Returns:
        Tuple of (x, y) coordinates of the foot position
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int(y2)