#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import cv2
import os
import pandas
import scipy
from scipy.stats import chi2

from object_detector_retinanet.keras_retinanet.utils.Boxes import BOX, extract_boxes_from_edge_boxes, \
    perform_nms_on_image_dataframe
from object_detector_retinanet.keras_retinanet.utils.CollapsingMoG import collapse
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr
from object_detector_retinanet.utils import root_dir


class Params:
    box_size_factor = 0.3
    min_box_size = 2
    ellipsoid_thresh = 0.5


def gaussian_blur(w, h):
    sigmaX = w / 2.
    sigmaY = h / 2.
    x = numpy.linspace(-(w / 2.), (w / 2.), w)
    y = numpy.linspace(-(h / 2.), (h / 2.), h)
    x, y = numpy.meshgrid(x, y)
    x /= numpy.sqrt(2.) * sigmaX
    y /= numpy.sqrt(2.) * sigmaY
    x2 = x ** 2
    y2 = y ** 2
    kernel = numpy.exp(- x2 - y2)
    return kernel


def aggregate_gaussians(sub_range, shape, width, height, confidence, boxes):
    shape = [int(x) for x in shape]
    heat_map = numpy.zeros(shape=shape, dtype=numpy.float64)
    for i in sub_range:
        curr_gaussian = gaussian_blur(width[i], height[i])
        cv2.normalize(curr_gaussian, curr_gaussian, 0, confidence[i], cv2.NORM_MINMAX)
        box = boxes[:, i]
        shape = heat_map[box[BOX.Y1]:box[BOX.Y2], box[BOX.X1]:box[BOX.X2]].shape
        heat_map[box[BOX.Y1]:box[BOX.Y2], box[BOX.X1]:box[BOX.X2]] += curr_gaussian.reshape(shape)
    return heat_map


class DuplicateMerger(object):
    visualizer = None

    def filter_duplicate_candidates(self, data, image):

        Params.box_size_factor = 0.5
        Params.min_box_size = 5
        Params.ellipsoid_thresh = 0.5
        Params.min_k = 0

        # TODO time optimization: split into initial clusters using gaussian information rather than heatmap contours
        heat_map = numpy.zeros(shape=[image.shape[0], image.shape[1], 1], dtype=numpy.float64)
        original_detection_centers = self.shrink_boxes(data, heat_map)

        cv2.normalize(heat_map, heat_map, 0, 255, cv2.NORM_MINMAX)
        heat_map = cv2.convertScaleAbs(heat_map)
        h2, heat_map = cv2.threshold(heat_map, 4, 255, cv2.THRESH_TOZERO)
        contours = cv2.findContours(numpy.ndarray.copy(heat_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = self.find_new_candidates(contours, heat_map, data, original_detection_centers, image)
        candidates = self.map_original_boxes_to_new_boxes(candidates, original_detection_centers)

        # TODO time optimization: parallelize contours/clusters resolvers.
        # TODO time optimization: convert numpy to tensorflow/keras
        best_detection_ids = {}
        filtered_data = pandas.DataFrame(columns=data.columns)
        for i, candidate in candidates.items():
            label = candidate['original_detection_ids']
            original_detections = data.ix[label]
            original_detections[
                'avg_score'] = 0.5 * original_detections.confidence + 0.5 * original_detections.hard_score
            best_detection_id = original_detections.avg_score.argmax()
            # best_detection_id = original_detections.confidence.argmax()
            # best_detection_id = original_detections.hard_score.argmax()
            best_detection = original_detections.ix[best_detection_id].copy()

            # The following code creates the median bboxes
            # original_detections = original_detections[original_detections.confidence > 0.5]
            # if original_detections.shape[0] > 0:
            #     w = original_detections['x2'] - original_detections['x1']
            #     h = original_detections['y2'] - original_detections['y1']
            #     x = original_detections['x1'] + 0.5 * w
            #     y = original_detections['y1'] + 0.5 * h
            #
            #     med_x = int(round(scipy.percentile(x, 50)))
            #     med_y = int(round(scipy.percentile(y, 50)))
            #     med_w = int(round(scipy.percentile(w, 50)))
            #     med_h = int(round(scipy.percentile(h, 50)))
            #     best_detection['x1'] = med_x - med_w / 2
            #     best_detection['y1'] = med_y - med_h / 2
            #     best_detection['x2'] = med_x + med_w / 2
            #     best_detection['y2'] = med_y + med_h / 2

            best_detection_ids[best_detection_id] = best_detection
            filtered_data = filtered_data.append(best_detection)

        # to handle overlap between contour bboxes
        filtered_data = perform_nms_on_image_dataframe(filtered_data, 0.3)

        return filtered_data

    def find_new_candidates(self, contours, heat_map, data, original_detection_centers, image):
        candidates = []
        for contour_i, contour in enumerate(contours[1]):
            contour_bounding_rect = cv2.boundingRect(contour)

            contour_bbox = extract_boxes_from_edge_boxes(numpy.array(contour_bounding_rect))[0]
            box_width = contour_bbox[BOX.X2] - contour_bbox[BOX.X1]
            box_height = contour_bbox[BOX.Y2] - contour_bbox[BOX.Y1]
            contour_area = cv2.contourArea(contour)
            offset = contour_bbox[0:2]
            mu = None
            cov = None
            original_indexes = self.get_contour_indexes(contour, contour_bbox, original_detection_centers['x'],
                                                        original_detection_centers['y'])
        
            n = original_indexes.sum()
            if n > 0 and box_width > 3 and box_height > 3:
                curr_data = data[original_indexes]
                w = (curr_data['x2'] - curr_data['x1']) * Params.box_size_factor
                h = (curr_data['y2'] - curr_data['y1']) * Params.box_size_factor
                areas = w * h
                median_area = areas.median()
                if median_area > 0:
                    approximate_number_of_objects = min(numpy.round(contour_area / median_area), 100)
                else:
                    approximate_number_of_objects = 0
                sub_heat_map = numpy.copy(heat_map[contour_bbox[BOX.Y1]:contour_bbox[BOX.Y2],
                                          contour_bbox[BOX.X1]:contour_bbox[BOX.X2]])
                k = max(1, int(approximate_number_of_objects))
                # print n,k
                if k >= 1 and n > k:
                    if k > Params.min_k:
                        beta, mu, cov = collapse(original_detection_centers[original_indexes].copy(), k, offset,
                                                 max_iter=20, epsilon=1e-10)
                    if mu is None:  # k<=Params.min_k or EM failed
                        print (n, k, ' k<=Params.min_k or EM failed')
                        self.perform_nms(candidates, contour_i, curr_data)
                    else:  # successful EM
                        cov, mu, num, roi = self.remove_redundant(contour_bbox, cov, k, mu, image, sub_heat_map)
                        self.set_candidates(candidates, cov, heat_map, mu, num, offset, roi, sub_heat_map)
                elif (k == n):
                    pass
                    # print n, k, ' k==n'
                    # self.perform_nms(candidates, contour_i, curr_data)

        return candidates

    def set_candidates(self, candidates, cov, heat_map, mu, num, offset, roi, sub_heat_map):
        for source_i, ((_x, _y), c) in enumerate(zip(mu, cov)):
            sigmax = numpy.sqrt(c[0, 0])
            sigmay = numpy.sqrt(c[1, 1])
            _x1 = int(round(max(0, _x - 2 * sigmax)))
            _y1 = int(round(max(0, _y - 2 * sigmay)))
            _x2 = int(round(min(sub_heat_map.shape[1], _x + 2 * sigmax)))
            _y2 = int(round(min(sub_heat_map.shape[0], _y + 2 * sigmay)))

            local_box = [_x1, _y1, _x2, _y2]
            abs_box = numpy.array(self.local_box_offset(offset, local_box))
            box_width = abs_box[BOX.X2] - abs_box[BOX.X1]
            box_height = abs_box[BOX.Y2] - abs_box[BOX.Y1]
            if box_width > Params.min_box_size and box_height > Params.min_box_size:
                candidates.append({'box': abs_box, 'original_detection_ids': [],
                                   'score': heat_map[abs_box[BOX.Y1]:abs_box[BOX.Y2],
                                            abs_box[BOX.X1]:abs_box[BOX.X2]].max()})

    def remove_redundant(self, contour_bbox, cov, k, mu, image, sub_heat_map):
        mu = mu.round().astype(numpy.int32)

        roi = image[contour_bbox[BOX.Y1]:contour_bbox[BOX.Y2],
              contour_bbox[BOX.X1]:contour_bbox[BOX.X2]].copy()
        cnts = []
        for source_i, ((_x, _y), c) in enumerate(zip(mu, cov)):
            sigmax = numpy.sqrt(c[0, 0])
            sigmay = numpy.sqrt(c[1, 1])

            chi_square_val = numpy.math.sqrt(chi2.ppf(Params.ellipsoid_thresh, 2))
            retval, eigenvalues, eigenvectors = cv2.eigen(c)
            angle = numpy.math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
            if angle < 0:
                angle += 2 * numpy.math.pi
            angle = 180 * angle / numpy.math.pi

            half_major_axis_size = chi_square_val * numpy.math.sqrt(eigenvalues[1])
            half_minor_axis_size = chi_square_val * numpy.math.sqrt(eigenvalues[0])

            local_m = numpy.zeros_like(sub_heat_map)
            poly = cv2.ellipse2Poly((int(round(_x)), int(round(_y))),
                                    (int(round(half_minor_axis_size)), int(round(half_major_axis_size))),
                                    -int(round(angle)), 0, 360, 15)
            ellipse_mask = cv2.fillPoly(local_m, [poly], (1, 1, 1))
            contours = cv2.findContours(ellipse_mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cnts.append(contours[1][0])
        center_points = mu.copy()
        distances = scipy.spatial.distance.cdist(center_points, center_points)
        scaled_distances = numpy.ndarray(shape=[k, k], dtype=numpy.float64)
        for i in range(0, k):
            for j in range(0, k):
                if i == j:
                    scaled_distances[i, j] = 0
                    continue
                cnt_i = cnts[i]
                cnt_j = cnts[j]
                ct_i_to_pt_j = -cv2.pointPolygonTest(cnt_i, (mu[j][0], mu[j][1]), measureDist=True)
                ct_j_to_pt_i = -cv2.pointPolygonTest(cnt_j, (mu[i][0], mu[i][1]), measureDist=True)
                if ct_i_to_pt_j <= 0 or ct_j_to_pt_i <= 0:
                    scaled_distances[i, j] = -numpy.inf
                else:
                    pt_dist = distances[i, j]
                    ct_i_to_ct_j = ct_i_to_pt_j - pt_dist + ct_j_to_pt_i
                    scaled_distances[i, j] = ct_i_to_ct_j
        scaled_distances = numpy.triu(scaled_distances)
        i_s, j_s = numpy.unravel_index(numpy.argsort(scaled_distances, axis=None), scaled_distances.shape)
        to_remove = []
        for i, j in zip(i_s, j_s):
            if scaled_distances[i, j] >= 0:
                break
            if i not in to_remove and j not in to_remove:
                pt_i = center_points[i]
                pt_j = center_points[j]
                pt1_x = max(0, min(pt_i[1], sub_heat_map.shape[0] - 1))
                pt1_y = max(0, min(pt_i[0], sub_heat_map.shape[1] - 1))
                pt2_x = max(0, min(pt_j[1], sub_heat_map.shape[0] - 1))
                pt2_y = max(0, min(pt_j[0], sub_heat_map.shape[1] - 1))

                val_i = sub_heat_map[pt1_x, pt1_y]
                val_j = sub_heat_map[pt2_x, pt2_y]
                remove_id = i
                if val_j < val_i:
                    remove_id = j
                to_remove.append(remove_id)
        if len(to_remove) > 0:
            to_remove = numpy.array(to_remove)
            mask = numpy.zeros(mu.shape[0])
            mask[to_remove] = 1
            mask = mask.astype(numpy.bool)
            mu = mu[~mask]
            cov = cov[~mask]
        num = mu.shape[0]
        return cov, mu, num, roi

    def perform_nms(self, candidates, contour_i, curr_data):

        nms_data = perform_nms_on_image_dataframe(curr_data, 0.3)

        for sub_ind, row in nms_data.iterrows():
            curr_box = numpy.asarray([row['x1'], row['y1'], row['x2'], row['y2']])
            box_width = curr_box[BOX.X2] - curr_box[BOX.X1]
            box_height = curr_box[BOX.Y2] - curr_box[BOX.Y1]
            if box_width > Params.min_box_size and box_height > Params.min_box_size:
                candidates.append({'box': curr_box, 'original_detection_ids': []})

    def get_contour_indexes(self, contour, contour_bbox, x, y):
        original_indexes = (contour_bbox[BOX.X1] <= x) & (x <= contour_bbox[BOX.X2]) & (
                contour_bbox[BOX.Y1] <= y) & (y <= contour_bbox[BOX.Y2])
        return original_indexes

    def local_box_offset(self, offset, box):
        box_offset = [0, 0, 0, 0]
        box_offset[BOX.X1] = box[BOX.X1] + offset[0]
        box_offset[BOX.Y1] = box[BOX.Y1] + offset[1]
        box_offset[BOX.X2] = box[BOX.X2] + offset[0]
        box_offset[BOX.Y2] = box[BOX.Y2] + offset[1]
        return box_offset


    def shrink_boxes(self, data, heat_map):
        x1 = data['x1']
        y1 = data['y1']
        x2 = data['x2']
        y2 = data['y2']

        width = x2 - x1
        height = y2 - y1
        original_detection_centers_x = x1 + width / 2.
        original_detection_centers_y = y1 + height / 2.
        original_detection_centers = original_detection_centers_x.to_frame('x').join(
            original_detection_centers_y.to_frame('y'))

        boxes = x1.to_frame('x1').join(x2.to_frame('x2')).join(y1.to_frame('y1')).join(y2.to_frame('y2'))
        w_shift = ((width * (1 - Params.box_size_factor)) / 2.).astype(numpy.int32)
        h_shift = ((height * (1 - Params.box_size_factor)) / 2.).astype(numpy.int32)

        boxes.x1 += w_shift
        boxes.x2 -= w_shift
        boxes.y1 += h_shift
        boxes.y2 -= h_shift

        width = boxes.x2 - boxes.x1
        height = boxes.y2 - boxes.y1
        confidence = data['confidence']

        original_detection_centers = original_detection_centers.join(boxes.x1.to_frame('left_x'))
        original_detection_centers = original_detection_centers.join(boxes.x2.to_frame('right_x'))
        original_detection_centers = original_detection_centers.join(boxes.y1.to_frame('top_y'))
        original_detection_centers = original_detection_centers.join(boxes.y2.to_frame('bottom_y'))
        original_detection_centers = original_detection_centers.join((width / 2.).to_frame('sigma_x'))
        original_detection_centers = original_detection_centers.join((height / 2.).to_frame('sigma_y'))
        original_detection_centers = original_detection_centers.join(confidence.to_frame('confidence'))

        confidence = numpy.array(confidence)
        width = numpy.asarray(width)
        height = numpy.asarray(height)
        boxes = numpy.asarray([boxes.x1.values, boxes.y1.values, boxes.x2.values, boxes.y2.values], dtype=numpy.int32)

        compression_factor = self.compression_factor
        orig_shape = heat_map.shape
        shape = (orig_shape[0] / compression_factor, orig_shape[1] / compression_factor, orig_shape[2])
        small_heat_map = heat_map
        if compression_factor > 1:
            width /= compression_factor
            height /= compression_factor
            boxes /= compression_factor

            width = numpy.round(width).astype(int)
            height = numpy.round(height).astype(int)
            boxes[BOX.X1] = numpy.maximum(numpy.round(boxes[BOX.X1]), 0)
            boxes[BOX.Y1] = numpy.maximum(numpy.round(boxes[BOX.Y1]), 0)
            boxes[BOX.X2] = numpy.minimum(boxes[BOX.X1] + width, shape[1])
            boxes[BOX.Y2] = numpy.minimum(boxes[BOX.Y1] + height, shape[0])

            small_heat_map = numpy.zeros(shape=shape, dtype=numpy.float64)

        small_heat_map += aggregate_gaussians(sub_range=range(0, data.shape[0]), shape=shape, width=width,
                                              height=height, confidence=confidence, boxes=boxes)
        if compression_factor > 1:
            heat_map += numpy.expand_dims(cv2.resize(small_heat_map, (orig_shape[1], orig_shape[0])), axis=2)
        return original_detection_centers

    def map_original_boxes_to_new_boxes(self, candidates, original_detection_centers):
        x = original_detection_centers['x']
        y = original_detection_centers['y']
        matched_indexes = numpy.ndarray(shape=original_detection_centers.shape[0], dtype=numpy.bool)
        matched_indexes.fill(False)
        for candidate in candidates:
            box = candidate['box']
            original_indexes = (box[BOX.X1] <= x) & (x <= box[BOX.X2]) & (box[BOX.Y1] <= y) & (
                    y <= box[BOX.Y2]) & ~matched_indexes
            matched_indexes[original_indexes] = True
            candidate['original_detection_ids'] = list(original_indexes[original_indexes].keys())

        new_candidates = {}
        i = 0
        for candidate in candidates:
            if len(candidate['original_detection_ids']) > 0:
                new_candidates[i] = candidate
                i += 1

        return new_candidates


def merge_detections(image_name, results):
#    project = 'SKU_dataset'
    result_df = pandas.DataFrame()
    result_df['x1'] = results[:, 0].astype(int)
    result_df['y1'] = results[:, 1].astype(int)
    result_df['x2'] = results[:, 2].astype(int)
    result_df['y2'] = results[:, 3].astype(int)
    result_df['confidence'] = results[:, 4]
    result_df['hard_score'] = results[:, 5]
    result_df['uuid'] = 'object_label'
    result_df['label_type'] = 'object_label'
#    result_df['project'] = project
    result_df['image_name'] = image_name

    result_df.reset_index()
    result_df['id'] = result_df.index
    pixel_data = None
    duplicate_merger = DuplicateMerger()
    duplicate_merger.multiprocess = False
    duplicate_merger.compression_factor = 1
#    project = result_df['project'].iloc[0]
    image_name = result_df['image_name'].iloc[0]
    if pixel_data is None:
        pixel_data = read_image_bgr(os.path.join(root_dir(),  image_name))

    filtered_data = duplicate_merger.filter_duplicate_candidates(result_df, pixel_data)
    return filtered_data


if __name__ == '__main__':
    merge_detections()
