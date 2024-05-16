import os
import torch 
import torch.utils.data as data
import numpy as np
import pyvista as pv
import random
import pytorch3d
from pytorch3d import ops

def get_dataset(args, set_type, datasets, train=True):
    if set_type == 'train' and train==True:
        train = True 
    else:
        train = False
    dataset_dict = get_dataset_dict(args, set_type, datasets)
    # 4D / spatiotemporal
    if '4d' in args and args['4d'] == True:
        dataset = PC_Sequence_Dataset(args, dataset_dict, train)
    # 3D
    else:
        if 'dpc' in args.model_name:
            dataset = Paired_PC_Dataset(args, dataset_dict, train)
        else:
            dataset = PC_Dataset(args, dataset_dict, train)
    return dataset

def get_dataset_dict(args, set_type, datasets):
    data_dirs = []
    for dataset in datasets:
        data_dirs.append(os.path.join('data/', dataset, set_type)+'/')

    subsample = args.train_subset_size
    if args.train_subset_size != None and args.train_subset_size > 1 and set_type=='train':
        subsample = args.train_subset_size
    else:
        subsample = None

    point_sets = []
    names = []
    labels = []
    i=0
    for data_dir in data_dirs:
        for file in sorted(os.listdir(data_dir))[:subsample]:
            if '.vtk' in file:
                points = np.array(pv.read(data_dir+file).points)
            elif '.particles' in file:
                points = np.loadtxt(data_dir+file)
            else:
                print("Error: unreconginzed file")
                break
            point_sets.append(points)
            names.append(file.replace(".vtk","").replace(".particles",""))
            labels.append(i)
        i += 1

    dataset_dict = {}
    dataset_dict['point_sets'] = point_sets
    dataset_dict['names'] = names 
    dataset_dict['labels'] = labels

    return dataset_dict

# Point cloud dataset
class PC_Dataset(data.Dataset):
    def __init__(self, args, dataset_dict, train=False):
        self.num_points = args.num_input_points
        self.missing_percent = args.missing_percent
        self.noise_level = args.noise_level
        if self.noise_level== None or self.noise_level==0:
            self.add_noise = False
        else:
            self.add_noise = True
        self.rot_range = args.rot_range
        if self.rot_range == None or self.rot_range==0 or not train:
            self.add_rot= False
        else:
            self.add_rot = True

        self.point_sets = dataset_dict['point_sets']
        self.names = dataset_dict['names']
        self.labels = dataset_dict['labels']

        self.train = train
        self.num_gt = 10000 #5000

    def __getitem__(self, index):
        full_point_set = self.point_sets[index]
        if self.add_rot:
            R = get_random_rot(self.rot_range)
            full_point_set = full_point_set @ R.T
        name = self.names[index]
        label = self.labels[index]

        # add missingness
        if not self.missing_percent or self.missing_percent == 0:
            partial_point_set = full_point_set
        else:
            if self.missing_percent == -1:
                missing_percent = np.random.uniform(0, 0.5)
            else:
                missing_percent = self.missing_percent
            if self.train:
                seed = np.random.randint(len(full_point_set))
            else:
                seed = 0 # consistent testing
            distances = np.linalg.norm(full_point_set - full_point_set[seed], axis=1)
            sorted_points = full_point_set[np.argsort(distances)]
            partial_point_set = sorted_points[int(len(full_point_set)*missing_percent):]

        # select subset
        if self.num_points > len(partial_point_set):
            replace = True
        else: 
            replace = False
        choice = np.random.choice(len(partial_point_set), self.num_points, replace=replace)
        partial = torch.FloatTensor(partial_point_set[choice, :])
        
        # add noise
        if self.add_noise:
            partial = partial + (self.noise_level)*torch.randn(partial.shape)
        
        # ground truth 
        choice = np.random.choice(len(full_point_set), self.num_gt, replace=True)
        gt = torch.FloatTensor(full_point_set[choice, :])

        return partial, gt, label, name

    def __len__(self):
        return len(self.point_sets)

# Point cloud dataset
class PC_Sequence_Dataset(data.Dataset):
    def __init__(self, args, dataset_dict, train=False):
        self.num_points = args.num_input_points
        self.num_time_points = args.num_time_points
        self.missing_percent = args.missing_percent
        self.noise_level = args.noise_level
        if self.noise_level== None or self.noise_level==0:
            self.add_noise = False
        else:
            self.add_noise = True
        self.rot_range = args.rot_range
        if self.rot_range == None or self.rot_range==0 or not train:
            self.add_rot= False
        else:
            self.add_rot = True

        point_sets = dataset_dict['point_sets']
        names = dataset_dict['names']
        labels = dataset_dict['labels']

        # Reshape into sequences
        self.point_set_seqs = [point_sets[i:i + self.num_time_points] for i in range(0, len(point_sets), self.num_time_points)] 
        self.names = [names[i:i + self.num_time_points] for i in range(0, len(names), self.num_time_points)] 
        self.labels = [labels[i:i + self.num_time_points] for i in range(0, len(labels), self.num_time_points)] 

        # Debug
        for i in range(len(self.names)):
            for j in range(self.num_time_points):
                name = self.names[i][j]
                pts = self.point_set_seqs[i][j]
                np.savetxt('debug/test/'+name+'.particles', pts)

    def __getitem__(self, index):
        point_seq = self.point_set_seqs[index]
        name = self.names[index]
        label = torch.tensor(self.labels[index])

        input_seq = torch.zeros((self.num_time_points, self.num_points, 3))
        gt_seq = torch.zeros((self.num_time_points, 5000, 3))
        for t in range(self.num_time_points):
            full_point_set = point_seq[t]

            choice = np.random.choice(len(full_point_set), 5000, replace=True)
            gt_seq[t] = torch.FloatTensor(full_point_set[choice, :])

            if self.add_rot:
                R = get_random_rot(self.rot_range)
                full_point_set = full_point_set @ R.T

            # add missingness
            if not self.missing_percent or self.missing_percent == 0:
                partial_point_set = full_point_set
            else:
                if self.missing_percent == -1:
                    missing_percent = np.random.uniform(0, 0.5)
                else:
                    missing_percent = self.missing_percent
                if self.train:
                    seed = np.random.randint(len(full_point_set))
                else:
                    seed = 0 # consistent testing
                distances = np.linalg.norm(full_point_set - full_point_set[seed], axis=1)
                sorted_points = full_point_set[np.argsort(distances)]
                partial_point_set = sorted_points[int(len(full_point_set)*missing_percent):]

            # select subset
            if self.num_points > len(partial_point_set):
                replace = True
            else: 
                replace = False
            choice = np.random.choice(len(partial_point_set), self.num_points, replace=replace)
            partial = torch.FloatTensor(partial_point_set[choice, :])
            
            # add noise
            if self.add_noise:
                partial = partial + (self.noise_level)*torch.randn(partial.shape)
            input_seq[t] = partial
        
        return input_seq, gt_seq, label, name

    def __len__(self):
        return len(self.point_set_seqs)

def get_random_rot(deg):
    deg = np.deg2rad(deg)
    theta_x = np.random.uniform(low=-1*deg, high=deg)
    theta_y = np.random.uniform(low=-1*deg, high=deg)
    theta_z = np.random.uniform(low=-1*deg, high=deg)
    R1 = np.eye(3)
    R1[1, 1] = np.cos(theta_x)
    R1[2, 2] = np.cos(theta_x)
    R1[1, 2] = -1*np.sin(theta_x)
    R1[2, 1] = np.sin(theta_x)
    R2 = np.eye(3)
    R2[0, 0] = np.cos(theta_y)
    R2[2, 2] = np.cos(theta_y)
    R2[2, 0] = -1*np.sin(theta_y)
    R2[0, 2] = np.sin(theta_y)
    R3 = np.eye(3)
    R3[1, 1] = np.cos(theta_z)
    R3[2, 2] = np.cos(theta_z)
    R3[1, 2] = -1*np.sin(theta_z)
    R3[2, 1] = np.sin(theta_z)
    R = np.matmul(np.matmul(R1, R2), R3)
    return R

# Paired point cloud dataset - DPC
class Paired_PC_Dataset(data.Dataset):
    def __init__(self, args, dataset_dict, train=False):
        self.num_points = args.num_input_points
        self.pc_dataset = PC_Dataset(args, dataset_dict, train)
        if not train:
            ref_points = np.array(pv.read(args.ref_path).points)
            target_pc = torch.FloatTensor(ref_points).to('cuda:0')
            target_pc, _ = pytorch3d.ops.sample_farthest_points(target_pc[None,:], torch.Tensor([self.num_points]).to('cuda:0'))
            self.target_pc = target_pc.squeeze()
        else:
            self.target_pc = None
        
    def __getitem__(self, index):
        source_pc, source_gt, source_label, source_name = self.pc_dataset.__getitem__(index)
        if self.target_pc == None:
            choices = list(range(0,index)) + list(range(index+1, len(self.pc_dataset.point_sets)))
            target_index = random.choice(choices)
            target_pc, target_gt, target_label, target_name = self.pc_dataset.__getitem__(target_index)
        else:
            target_pc = self.target_pc
        return source_pc, target_pc, source_gt, source_label, source_name

    def __len__(self):
        return len(self.pc_dataset.point_sets)