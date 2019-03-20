import keras
import sys

import numpy

from object_detector_retinanet.keras_retinanet.layers import RegressBoxes, ClipBoxes, \
    FilterDetections
from object_detector_retinanet.keras_retinanet.models.retinanet import AnchorParameters, \
    retinanet, __build_anchors


def retinanet_iou(
        model=None,
        anchor_parameters=AnchorParameters.default,
        name='retinanet-iou',
        **kwargs
):
    if model is None:
        model = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    BasicModel.freeze_base_layers(model.layers)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors = __build_anchors(anchor_parameters, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    # boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # # filter detections (apply NMS / score threshold / select top-k)
    # detections = FilterDetections(nms=False, name='filtered_detections')(
    #     [boxes, classification] + other)

    # concat = keras.layers.Concatenate([detections[0], detections[-1]], name='iou_regression_output')
    concatenation_layer = keras.layers.Concatenate(axis=-1, name='iou_regression_output')
    concatenated_top = concatenation_layer([boxes, other[0], classification])
    iou_model = keras.models.Model(inputs=model.inputs, outputs=concatenated_top, name=name)

    return iou_model


class BasicModel(object):
    @staticmethod
    def freeze_base_layers(layers):
        for layer in layers:
            if 'iou' not in layer.name:
                layer.trainable = False
            if hasattr(layer, 'layers'):
                BasicModel.freeze_base_layers(layer.layers)
