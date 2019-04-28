This is *NOT* the paper web page but just an internal code repo for us. 

## Notes
This implementation is built on top of https://github.com/fizyr/keras-retinanet.

The SKU110K dataset is provided in csv format compatible with the code CSV parser.

The output files will be saved under "$HOME"/Documents/SKU110K and have the same structure as in https://github.com/fizyr/keras-retinanet.

This repository requires Keras 2.2.4 or higher.

This repository is tested using Python 2.7.6 and OpenCV 3.1.

Note: The detector here is a stronger version of the one we originally used, and can thus achieve
higher results than the ones we originally reported.

The EM-merger provided here is the stable version (not time-optimized). Some of the changes required for
optimization are mentioned in the TO-DO comments.

Contributions to this project are welcome.

## Usage

move the unzipped SKU100K folder to "$HOME"/Documents

set $PYTHONPATH to the repository root e.g. "/home/ubuntu/dev/SKU110K"

train:

(1) Train the base model:
python -u object_detector_retinanet/keras_retinanet/bin/train.py csv

The weight h5 files will be saved in the "snapshot" folder

(2) train the IoU layer:

Select and copy a "WEIGHT_FILE" file from step (1) to "$HOME"/Documents/SKU100K/ and run
python -u object_detector_retinanet/keras_retinanet/bin/train_iou.py --WEIGHT_FILE csv

e.g.:
python -u object_detector_retinanet/keras_retinanet/bin/train_iou.py --gpu 0 --weights resnet50_csv_10.h5 csv > train_iou_sku110k.log &


(3) evaluate:
Select and copy a "WEIGHT_FILE" file from step (2) to "$HOME"/Documents/SKU100K/ and run


python -u object_detector_retinanet/keras_retinanet/bin/evaluate_iou.py csv WEIGHT_FILE

e.g:
nohup env PYTHONPATH="/home/ubuntu/dev/SKU110K" python -u object_detector_retinanet/keras_retinanet/bin/evaluate_iou.py --gpu 3 csv iou_resnet50_csv_07.h5 --force_hard_score=True> eval_iou_sku110k.log &



