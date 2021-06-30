# import keras
import keras

# import keras_retinanet
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from object_detector_retinanet.keras_retinanet.utils.colors import label_color

# import for EM Merger and viz
from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_folder, root_dir


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join
import shutil as sh
import numpy as np
import time
from tqdm import tqdm
from PIL import Image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


class load_our_config:
    """
    Hard Code your data here (input data to initialize)
    """
    def __init__(self):
        ### img_path or folder
        self.img_dir = '/home/ubuntu/Documents/SKU110K/beers/'
        self.predictions_dir = join(self.img_dir, 'predictions')        
        ### delete old predictions if the dir exists
        try:
            sh.rmtree(self.predictions_dir)
        except:
            pass

        self.image_name = False # 'PIWA_AR3006_F00007_468943_PIWO 1.jpg' # set to False to use whole directory
        if self.image_name:
            self.images_paths = [join(self.img_dir, self.image_name)]  # add it as list so we can iterate despite format
        else:
            ### it's a directory. get files there (be sure to put only images!)
            self.images_paths = [join(self.img_dir,x) for x in os.listdir(self.img_dir)]
        print("self.images_paths: ", self.images_paths)
        
        ### create a new folder for predicted images
        os.mkdir(self.predictions_dir)
        self.model_path = '/home/ubuntu/Documents/iou_resnet50_csv_06.h5'
        self.backbone_name = 'resnet50'
        self.hard_score_rate=.3
        self.max_detections = 9999
        # for filtering predictions based on score (objectness/confidence)
        self.threshold = 0.3
        # load label to names mapping for visualization purposes
        self.labels_to_names = {0: 'object'}



def main(our_config):
    ### Load Retinanet + IoU Model
    our_config.model = models.load_model(our_config.model_path, backbone_name=our_config.backbone_name,
        convert=1, nms=False)
    # model = models.load_model(model_path, backbone_name='resnet50')

    for our_config.image_path in tqdm(our_config.images_paths):
        # load image
        our_config.image = read_image_bgr(our_config.image_path)
        predict(our_config)

def predict(our_config):
    # copy to draw on
    draw = our_config.image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(our_config.image)
    image, scale = resize_image(image)
    
    # Run inference
    boxes, hard_scores, labels, soft_scores = our_config.model.predict_on_batch(np.expand_dims(image, axis=0))
    #########################
    soft_scores = np.squeeze(soft_scores, axis=-1)
    soft_scores = our_config.hard_score_rate * hard_scores + (1 - our_config.hard_score_rate) * soft_scores
    # correct boxes for image scale
    boxes /= scale
    # select indices which have a score above the threshold
    indices = np.where(hard_scores[0, :] > our_config.threshold)[0]
    
    # select those scores
    scores = soft_scores[0][indices]
    hard_scores = hard_scores[0][indices]
    
    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:our_config.max_detections]
    
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
    filtered_data = EmMerger.merge_detections(our_config.image_path, results)
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score'])
    
    for ind, detection in filtered_data.iterrows():
        box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
        filtered_boxes.append(box)
        filtered_scores.append(detection['confidence'])
        filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
        row = [our_config.image_path, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
               detection['confidence'], detection['hard_score']]
        csv_data_lst.append(row)
     
    
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        # scores are sorted so we can break
        if score < our_config.threshold:
            break
            
        color = [31, 0, 255] #label_color(label) ## BUG HERE LABELS ARE FLOATS SO COLOR IS HARDCODED 
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(our_config.labels_to_names[0], score)
        draw_caption(draw, b, caption)
    
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(draw)
    base_name = os.path.split(our_config.image_path)[1]
    predicted_img_path = join(our_config.predictions_dir, base_name)
    plt.savefig(predicted_img_path, bbox_inches='tight', pad_inches=0, dpi=220)


if __name__ == '__main__':
    our_config = load_our_config()
    # print(config.model_path)
    main(our_config)
