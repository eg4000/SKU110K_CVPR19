import numpy

X1 = 'x1'
X2 = 'x2'
Y1 = 'y1'
Y2 = 'y2'
BOX_CONSTANTS = [X1, Y1, X2, Y2]

class BOX:
    def __init__(self):
        pass

    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3
    X1_LEFT = 0
    Y1_LEFT = 1
    X2_LEFT = 2
    Y2_LEFT = 3
    X1_RIGHT = 4
    Y1_RIGHT = 5
    X2_RIGHT = 6
    Y2_RIGHT = 7

def reshape_vector(ndarr):
    """
    :param ndarr: take list and transform it to a ndarray with reshape
    :return: numpy array of numpy array
    """
    if not isinstance(ndarr, numpy.ndarray):
        # If ndarr is not a ndarray raise exception
        msg = 'This is not a ndarray type: type{}'.format(type(ndarr))
        raise TypeError(msg)

    if len(ndarr.shape) == 1:
        if len(ndarr) == 0:
            print('ndarray is empty, will not reshape')
            return ndarr
        ndarr_mat = ndarr.copy()
        ndarr_mat.resize(1, ndarr.size)
        return ndarr_mat
    return ndarr

def extract_boxes_from_edge_boxes(edge_boxes):
    edge_boxes = reshape_vector(edge_boxes)

    boxes = edge_boxes.copy()
    boxes[:, BOX.X2] = edge_boxes[:, 0] + edge_boxes[:, 2]
    boxes[:, BOX.Y2] = edge_boxes[:, 1] + edge_boxes[:, 3]
    return boxes

def box_area(boxes):
    """
    Calculates a box or boxes area.
    :param boxes: A list of boxes or a box (dictionary with keys x1, x2, y1, y2).
    :rtype: np.ndarray
    """
    boxes = reshape_vector(boxes)
    area_value = (boxes[:, BOX.X2] - boxes[:, BOX.X1]) * (boxes[:, BOX.Y2] - boxes[:, BOX.Y1])

    return area_value


def intersection(boxes, candidate_box):
    """
    Calculates the intersection  of a given box and an array of boxes.
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is width, boxes[3] is height.
    :return: intersection vector
    """

    boxes = reshape_vector(boxes)

    intersection_value = \
        numpy.maximum(0, numpy.minimum(boxes[:, BOX.X2], candidate_box[BOX.X2]) -
                      numpy.maximum(boxes[:, BOX.X1], candidate_box[BOX.X1])) * \
        numpy.maximum(0, numpy.minimum(boxes[:, BOX.Y2], candidate_box[BOX.Y2]) -
                      numpy.maximum(boxes[:, BOX.Y1], candidate_box[BOX.Y1]))

    return intersection_value

def maximum_overlap(boxes, candidate_box):
    """
    Calculates the maximum overlap of a given box and an array of boxes.
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is 'x2', boxes[3] is 'y2'.
    :return: maximum overlap ratios vector
    """

    boxes = reshape_vector(boxes)

    intersection_value = intersection(boxes, candidate_box)

    candidate_area = box_area(candidate_box)
    boxes_area = box_area(boxes)
    minimum_area = numpy.minimum(boxes_area, candidate_area)

    indices = minimum_area > 0
    minimum_area_divide = minimum_area * indices + (1 - indices)
    max_overlap_value = (intersection_value * indices) / numpy.cast['float32'](minimum_area_divide)

    return max_overlap_value



def non_maximal_suppression(boxes, scores=None, labels=None, overlap_threshold=0.5):

    if scores is None:
        scores = numpy.ones(boxes.shape[0])

    if labels is None:
        labels = numpy.ones(boxes.shape[0])

    # values = numpy.sort(scores)
    # argsort - decreasing order (largest last)!
    indices = numpy.argsort(scores)

    nms_boxes, nms_scores, nms_labels, nms_predictions, deleted_indices  = [], [], [], [], []

    while indices.shape[0] > 0:
        best_confidence_sorted_index = indices.shape[0] - 1
        best_confidence_index = indices[best_confidence_sorted_index]

        # Saving the best confidence box (it is not suppressed).
        max_box = boxes[best_confidence_index]  # Another option, takes the avg: numpy.average(overlapping_boxes, 0)
        max_scores = scores[best_confidence_index]
        nms_boxes.append(max_box)
        nms_scores.append(max_scores)
        indices = numpy.delete(indices, best_confidence_sorted_index)
        if indices.shape[0] == 1:
            break

        overlap = maximum_overlap(boxes[indices[0:best_confidence_sorted_index]], boxes[best_confidence_index, :])

        if overlap is not None:
            deleted_indices += list(indices[overlap >= overlap_threshold])
            # Suppressing non maximal boxes.
            indices = indices[overlap < overlap_threshold]

    return numpy.asarray(nms_boxes), numpy.asarray(nms_scores), deleted_indices


def perform_nms_on_image_dataframe(image_data, overlap_threshold=0.5):
    number_of_images = len(image_data['image_name'].unique())
    if number_of_images > 1:
        print('nms received data including more than 1 image - cannot perform nms!')
    image_boxes = image_data.as_matrix(BOX_CONSTANTS)
    image_scores = numpy.array(image_data['confidence'])

    nms_boxes, nms_scores, deleted_indices = non_maximal_suppression(image_boxes, image_scores,
                                                                     overlap_threshold=overlap_threshold)

    deleted_ids = list(image_data['id'].iloc[deleted_indices])
    return image_data[~image_data['id'].isin(deleted_ids)]