This is *NOT* the project web page. 

*Code has not been tested*

## Notes

This implementation is built on top of https://github.com/fizyr/keras-retinanet.
The SKU110K dataset is provided in csv format compatible with the code CSV parser.

Dependencies include: 'keras', 'keras-resnet', 'six', 'scipy'. 'Pillow', 'pandas', 'tensorflow-gpu'
This repository requires Keras 2.2.4 or higher, and was tested using Python 2.7.6 and OpenCV 3.1.

The output files will be saved under "$HOME"/Documents/SKU110K and have the same structure as in https://github.com/fizyr/keras-retinanet:
The weight h5 files will are saved in the "snapshot" folder and the tensorboard log files are saved in the "logs" folder.

Note that we have made several upgrades to the baseline detector since the beginning of this research, so the latest version can actually
achieve even higher results than the ones originally reported.

The EM-merger provided here is the stable version (not time-optimized). Some of the changes required for
optimization are mentioned in the TO-DO comments.

Contributions to this project are welcome.

## Usage

Move the unzipped SKU100K folder to "$HOME"/Documents

Set $PYTHONPATH to the repository root e.g. "/home/ubuntu/dev/SKU110K"

train:

(1) Train the base model:
python -u object_detector_retinanet/keras_retinanet/bin/train.py csv

(2) train the IoU layer:

python -u object_detector_retinanet/keras_retinanet/bin/train_iou.py --WEIGHT_FILE csv
where WEIGHT_FILE is the full path to the h5 file from step (1)

e.g.:
python -u object_detector_retinanet/keras_retinanet/bin/train_iou.py --gpu 0 --weights "/home/ubuntu/Documents/SKU110K/snapshot/Thu_May__2_17:07:11_2019/resnet50_csv_10.h5" > train_iou_sku110k.log &


(3) predict:

python -u object_detector_retinanet/keras_retinanet/bin/predict.py csv WEIGHT_FILE [--hard_score_rate=RATE]
where WEIGHT_FILE is the full path to the h5 file from step (2), and 0<=RATE<=1 computes the confidence as a weighted average between soft and hard scores. 

e.g:
nohup env PYTHONPATH="/home/ubuntu/dev/SKU110K" python -u object_detector_retinanet/keras_retinanet/bin/evaluate_iou.py --gpu 3 csv "/home/ubuntu/Documents/SKU110K/snapshot/Thu_May__2_17:10:30_2019/iou_resnet50_csv_07.h5" --hard_score_rate=0.5 > predict_sku110k.log &


The results are saved in CSV format in the "results" folder and drawn in "res_images_iou" folder.
