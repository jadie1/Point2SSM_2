# Point2SSM++
Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds

The run training by calling 'train.py' with a specificed config file, for example:
```
python train.py -c cfgs/point2ssm++.yaml
```
This will write the model, logged info, and a copy of the config file to a folder in `experiments/`, such as `experiments/spleen_all/point2ssm++_cd_l2_dgcnn/`.

To run inference, call `consist_test.py` with the config file and dataset, for example:
```
python consist_test.py -c experiments/spleen_all/point2ssm++_cd_l2_dgcnn/point2ssm++.yaml -d spleen
```
This will write the predicted correspondence points to the experiment directory in `output/`. 
