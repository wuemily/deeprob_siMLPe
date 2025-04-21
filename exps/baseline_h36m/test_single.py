import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    # for all motion_inputs and motion_targets TODO: where do we get motion_target??
    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        # print("banana", motion_input.shape)
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                # print("motion_input_ shape", motion_input_.shape)
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

# modifications:

from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

def process_data(filename):
    print(filename)
    info = open(filename, 'r').readlines() # opens file
    pose_info = [] 

    # remove commas, append to pose_info
    for line in info:
        line = line.strip().split(',')
        if len(line) > 0:
            pose_info.append(np.array([float(x) for x in line]))
            
    pose_info = np.array(pose_info) # turn into np array, [x, num_ppl * 75]

    pose_info = pose_info[:, 0:75] # take first human
    pose_info = pose_info.reshape(-1, 25, 3) # into [x, num_joints, 3]

    # BODY_25 indices that can be mapped to h36 32 joints
    body25_indices = np.array([
        10, 11, 12, 13, 15, 18, 1, 0,
        5, 6, 7, 2, 3, 4, 17, 20,
        15, 18, 19, 21, 22, 19
    ])

    # Corresponding H36M-style joint indices (matches joint_used_xyz)
    pose32_target_indices = np.array([
        2,  3,  4,  5,  7,  8,  9, 10,
        12, 13, 14, 15, 17, 18, 19, 21,
        22, 25, 26, 27, 29, 30
    ])

    pose_32 = np.zeros((pose_info.shape[0], 33, 3), dtype=np.float32) # get zeros for h36 format
    pose_32[:, pose32_target_indices, :] = pose_info[:, body25_indices, :] # set corresponding 

    pose_32[:, :2] = 0 # set first two joints to zero?? 

    N = pose_32.shape[0] 
    N = 50 # CHANGE LATER

    print('bananananana', pose_32.shape)
    pose_32 = pose_32[0:50, :, :] # shaping here for now, change later
    pose_info = pose_32.reshape(-1, 3) # reshape into (x*33, 3)
    print('dragonfruit', pose_info.shape)

    # def expmap2rotmat_torch(r):
    # """
    # Converts expmap matrix to rotation
    # batch pytorch version ported from the corresponding method above
    # :param r: N*3
    # :return: N*3*3

    # converts to rotation and then to xyz?
    pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(N, 33, 3, 3)[:, 1:]
    pose_info = rotmat2xyz_torch(pose_info)

    # sample rate stuff (take every 2 poses)
    sample_rate = 1 
    sampled_index = np.arange(0, N, sample_rate)
    h36m_motion_poses = pose_info[sampled_index]

    T = h36m_motion_poses.shape[0]
    h36m_motion_poses = h36m_motion_poses.reshape(T, 32, 3)

    return h36m_motion_poses


# end modifications
    # Example usage:
    # pred_xyz: shape (10, 22, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(config, model, dataloader))


    # new stuff
    print("NEW STUFFFF -------------------")

    input_filename = '/home/eric/Downloads/DeepRob/simlpe_repo/deeprob_siMLPe/data/single_test.txt'

    # dataloader = DataLoader(live_data, batch_size=128,
    #                         num_workers=1, drop_last=False,
    #                         sampler=sampler, shuffle=shuffle, pin_memory=True)
    
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)

    while(True):
        live_data = process_data(input_filename)
        data = live_data[:, joint_used_xyz, :].reshape(live_data.shape[0], -1).unsqueeze(0)
        data = data.cuda()

        print(model(data).reshape(1, 50, -1, 3).shape)

        prediction = model(data)[:, :25]
        pred_xyz = prediction.reshape(25, 22, 3)

        print('prediction output', prediction.shape)
        # write data to output text file


    
    # print("hi", live_data.shape)
    # dataloader = DataLoader(live_data, batch_size=128,
    #                         num_workers=1, drop_last=False,
    #                         sampler=sampler, shuffle=shuffle, pin_memory=True)
    
    # joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    # for (motion_input) in dataloader:
    #     print("initial motion input shape", motion_input.shape)
    #     # 128 batch size, 50 input poses, 32x2 joint positions
    #     motion_input = motion_input.cuda().unsqueeze(0)
    #     b,n,c,_ = motion_input.shape

    #     motion_input = motion_input.reshape(b, n, 32, 3)
    #     motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
    #     print(motion_input.shape) #torch.Size([128, 50, 32, 3])
    #     print("FORWARDS ", model.forward(motion_input).shape)

    # print(test(config, model, dataloader))

    # model.forward()

