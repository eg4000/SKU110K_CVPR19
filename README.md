# SKU-110K
Dataset and Codebase for CVPR2019 "Precise Detection in Densely Packed Scenes" [[Paper link]](https://arxiv.org/pdf/1904.00853.pdf)

<!---[alt text](figures/teaser_width.jpg)--->
<img src="figures/teaser_width.jpg" width="750">

A typical image in our SKU-110K, showing densely packed objects. (a) Detection results for the state-of-the-art RetinaNet[2], showing incorrect and overlapping detections, especially for the dark objects at the bottom which are harder to separate. (b) Our results showing far fewer misdetections and better fitting bounding boxes. (c) Zoomed-in views for RetinaNet[2] and (d) our method.


### Our novel contributions are:
1. **Soft-IoU layer**, added to an object detector to estimate the Jaccard index between the detected box and the (unknown) ground truth box.
2. **EM-Merger unit**, which converts detections and SoftIoU scores into a MoG (Mixture of Gaussians), and resolves overlapping detections in packed scenes.
3. **A new data set and benchmark**, the store keeping unit, 110k categories (SKU-110K), for item detection in store shelf images from around the world.

## Introduction
In our SKU-110K paper[1] we focus on detection in densely packed scenes, where images contain many objects, often looking similar or even identical, positioned in close proximity. These scenes are typically man-made, with examples including retail shelf displays, traffic, and urban landscape images. Despite the abundance of such environments, they are under-represented in existing object detection benchmarks, therefore, it is unsurprising that state-of-the-art object detectors are challenged by such images.


## Method
We propose learning the Jaccard index with a soft Intersection over Union (Soft-IoU) network layer. This measure provides valuable information on the quality of detection boxes. Those detections can be represented as a Mixture of Gaussians (MoG), reflecting their locations and their Soft-IoU scores. Then, an Expectation-Maximization (EM) based method is then used to cluster these Gaussians into groups, resolving detection overlap conflicts. 

## Dataset

<img src="figures/benchmarks_comparison.jpg" width="750">

We compare between key properties for related benchmarks. **#Img.**: Number of images. **#Obj./img.**: Average items per image. **#Cls.**: Number of object classes (more implies a harder detection problem due to greater appearance variations). **#Cls./img.**: Average classes per image. **Dense**: Are objects typically densely packed together, raising potential overlapping detection problems?. **Idnt**: Do images contain multiple identical objects or hard to separate object sub-regions?. **BB**: Bounding box labels available for measuring detection accuracy?.

## Qualitative Results
Add few results from the paper
<img src="qualitive_results.png" width="750">

## Dependencies

## Usage

## References
[1] Eran Goldman*, Roei Herzig*, Aviv Eisenschtat*, Jacob Goldberger, Tal Hassner, [Precise Detection in Densely Packed Scenes](https://arxiv.org/abs/1904.00853), 2019.

[2] Tsung-Yi Lin, Priyal Goyal, Ross Girshick, Kaiming He, Piotr Dollar, [Focal loss for dense object detection](https://arxiv.org/abs/1708.02002), 2018.


## Citation

```
@inproceedings{goldman2019dense,
 author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
 title     = {Precise Detection in Densely Packed Scenes},
 booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
 year      = {2019}
}
```
