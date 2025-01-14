def scale_boxes(found: list[tuple[float, float, float, float]] = [], *, window_size: int) -> list[tuple[float, float, float, float, float]]:
    scaled_boxes: list[tuple[float, float, float, float, float]] = []

    for scale, x, y, confidence in found:
        scaled_boxes.append((x * scale, y * scale, (x + window_size) * scale, (y + window_size) * scale, confidence))

    return scaled_boxes

def calculate_iou(box1: list, box2: list) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): The first bounding box in the format [xmin, ymin, xmax, ymax].
        box2 (list): The second bounding box in the format [xmin, ymin, xmax, ymax].

    Returns:
        float: The IoU of the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


def non_max_suppression(boxes: list, 
                        scores: list, 
                        iou_threshold: float) -> tuple[list, list]:
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    Args:
        boxes (list): A list of bounding boxes in the format [[xmin, ymin, xmax, ymax], ...].
        scores (list): A list of corresponding scores.
        iou_threshold (float): The IoU (Intersection over Union) threshold for merging bounding boxes.

    Returns:
       list: A list of indices of the boxes to keep after non-max suppression.
    """
    num_boxes = len(boxes)
    order = sorted(range(num_boxes), key=lambda i: scores[i], reverse=True)
    selected_indices = []
    summed_confidences = []

    while order:
        i = order.pop(0)
        selected_indices.append(i)
        sum_confidence = scores[i]
        to_remove = []
        for j in order:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                sum_confidence += scores[j]
                to_remove.append(j)
        order = [j for j in order if j not in to_remove]
        summed_confidences.append(sum_confidence)

    return selected_indices, summed_confidences

def compute_box_overlap(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """
    Compute the overlap score between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box in the format (xmin, ymin, xmax, ymax).
        box2 (tuple): The second bounding box in the format (xmin, ymin, xmax, ymax).

    Returns:
        float: The overlap score between the two bounding boxes.
    """
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    inter_width = max(0, inter_xmax - inter_xmin + 1)
    inter_height = max(0, inter_ymax - inter_ymin + 1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0