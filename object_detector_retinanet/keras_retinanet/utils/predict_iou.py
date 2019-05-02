"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import csv

import datetime

from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_folder, root_dir
from .visualization import draw_detections, draw_annotations

import numpy as np
import os

import cv2


def predict(
        generator,
        model,
        score_threshold=0.05,
        max_detections=9999,
        save_path=None,
        hard_score_rate=1.):
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score'])
    result_dir = os.path.join(root_dir(), 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()
    res_file = result_dir + '/detections_output_iou_{}_{}.csv'.format(hard_score_rate, timestamp)
    for i in range(generator.size()):
        image_name = os.path.join(generator.image_path(i).split(os.path.sep)[-2],
                                  generator.image_path(i).split(os.path.sep)[-1])
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, hard_scores, labels, soft_scores = model.predict_on_batch(np.expand_dims(image, axis=0))
        soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(hard_scores[0, :] > score_threshold)[0]

        # select those scores
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_hard_scores = hard_scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        results = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
             np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(image_name, results)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for ind, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                   detection['confidence'], detection['hard_score']]
            csv_data_lst.append(row)

        if save_path is not None:
            create_folder(save_path)

            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
                            np.asarray(filtered_labels), color=(0, 0, 255))

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    # Save annotations csv file
    with open(res_file, 'wb') as fl_csv:
        writer = csv.writer(fl_csv)
        writer.writerows(csv_data_lst)
    print("Saved output.csv file")
