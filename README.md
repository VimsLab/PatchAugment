# PatchAugment

UPDATE: Pre-trained models of ablation study and DGCNN code will be released SOON...

## Classification
This code submission is to reproduce 3D Shape Classification results of PointNet++ with PatchAugment on ModelNet40 dataset.

## Download Code and Unzip
unzip PatchAugment.zip
cd PatchAugment

## Environments
Ubuntu 18.04 <br>
Python 3.8.3 <br>
Pytorch 1.7.0

### Dataset
The train script downloads the ModelNet40 dataset (make sure it is in data folder) <br>
https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip<br>

### Run
```
Training
--------
python train_cls.py --model pointnet2_cls_ssg --log_dir log

uncomment layer in the model file for full model training

Testing
-------
python test_cls.py --log_dir logSA
```

### Performance
3D classification accuracy on PointNet++ single SA with our PatchAugment is 93.0%

## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)

## Note
Note: This code has been heaviy borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Also, the original code only includes scaling and translation from conventional data augmentation(using the aligned modelnet40 dataset). Hence the reported accuracies are higher. With all DA the accuracy drops(classification without normals on SSG model). However, in our method PatchAugment boosts the overall accuracy using all Data Augmentations at patch level instead of object level.


To cite our paper please use below bibtex.
  
```BibTex
  @inproceedings{sheshappanavar2021patchaugment,
    title={PatchAugment: Local Neighborhood Augmentation in Point Cloud Classification},
    author={Sheshappanavar, Shivanand Venkanna and Singh, Vinit Veerendraveer and Kambhamettu, Chandra},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={2118--2127},
    year={2021}
  }
```
