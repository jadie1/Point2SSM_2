# Point2SSM++
Implementation of [Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds](https://arxiv.org/abs/2405.09707).  If using this code, please cite the paper.

Run training by calling 'train.py' with a specificed config file, for example:
```
python train.py -c cfgs/point2ssm++.yaml
```
This will write the model, logged info, and a copy of the config file to a folder in `experiments/`, such as `experiments/spleen_all/point2ssm++_cd_l2_dgcnn/`.

To run inference, call `consist_test.py` with the config file and dataset, for example:
```
python consist_test.py -c experiments/spleen_all/point2ssm++_cd_l2_dgcnn/point2ssm++.yaml -d spleen
```
This will write the predicted correspondence points to the experiment directory, for example `experiments/spleen_all/point2ssm++_cd_l2_dgcnn/spleen/test/output/`. 


See `cfgs/point2ssm++_4d.yaml` for an example with 4D/spatiotemporal data, and `cfgs/point2ssm++_classifier.yaml` for multi-anatomy data. 

## Acknowledgements
This code utilizes the following Pytorch 3rd-party libraries and models:
- [Chamfer Distance](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html)
- [PointNet2](https://github.com/sshaoshuai/Pointnet2.PyTorch)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [PointAttN (SFA Block)](https://github.com/ohhhyeahhh/PointAttN)
- [PSTNet2](https://github.com/hehefan/PSTNet2)
