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


from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}

# BODY_25 to H36M joint mapping with comments
# ngl this is very sus cuz im not 100% sure about these
body25_indices = np.array([
    10, 11, 24, 22,     # Right leg: RKnee, RAnkle, RBigToe, RHeel
    13, 14, 21, 19,     # Left leg:  LKnee, LAnkle, LBigToe, LHeel
     8,  1,  0, 15,     # Torso/Head: MidHip, Neck, Nose, HeadTop
     5,  6,  7,         # Left arm: LShoulder, LElbow, LWrist
     7,  7,             # left arm hand deets
     2,  3,  4,         # Right arm: Rshould, relbow, rwrist
     4,  4              # r hand hand deeets
])

pose32_target_indices = np.array([
    2,  3,  4,  5,     # Right leg: RKnee0, RFoot1, # assumed  RToe2, RHeel3
    7,  8,  9, 10,     # Left leg:  LKnee4, LFoot5, # assumed LToe6, LHeel7
    12, 13, 14, 15,     # Torso/Head: SpineBase8, Thorax9, Neck10, Head11
    17, 18, 19,         # Left arm: LShoulder12, LElbow13, LWrist14
    21, 22,             # left arm hand details assumed (hand, finger) 15, 16
    25, 26, 27,         # R shoulder17, R elbow18, R wrist19
    29, 30              # R hand details assumed (hand20, finger21)
])

# H36M_NAMES = ['']*32 H36M_NAMES[0]  = 'Hip' H36M_NAMES[1]  = 'RHip' H36M_NAMES[2]  = 'RKnee' H36M_NAMES[3]  = 'RFoot'
# H36M_NAMES[6]  = 'LHip' H36M_NAMES[7]  = 'LKnee' H36M_NAMES[8]  = 'LFoot' H36M_NAMES[12] = 'Spine' H36M_NAMES[13] = 'Thorax'
# H36M_NAMES[14] = 'Neck/Nose' H36M_NAMES[15] = 'Head' H36M_NAMES[17] = 'LShoulder' H36M_NAMES[18] = 'LElbow'H36M_NAMES[19] = 'LWrist'
# H36M_NAMES[25] = 'RShoulder' H36M_NAMES[26] = 'RElbow' H36M_NAMES[27] = 'RWrist'

joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)

import matplotlib.pyplot as plt

# Define connections for 2D skeleton (based on H36M joint indices)
POSE22_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),       # Right leg: RKnee → RAnkle → RToe → RHeel
    (0, 8), (4, 8),               # knees to spinebase
    (4, 5), (5, 6), (6, 7),       # Left leg: LKnee → LAnkle → LToe → LHeel
    (8, 9), (9,10), (10,11),      # Spine → Thorax → Neck → Head
    (9,12), (12,13), (13,14),     # Left arm: Thorax → LShoulder → LElbow → LWrist
    (9,17), (17,18), (18,19),     # Right arm: Thorax → RShoulder → RElbow → RWrist
    (19,20), (20,21),             # Right fingers
    (14,15), (15,16)              # Left fingers
]

# Function to display one 2D pose
def show_pose_2d(joints_2d, title="Predicted Pose"):
    import matplotlib.pyplot as plt
    plt.clf()

    # Define joint region color mapping
    color_map = {
        "right_leg": "deepskyblue",
        "left_leg": "royalblue",
        "spine": "black",
        "head": "darkorange",
        "left_arm": "forestgreen",
        "right_arm": "seagreen",
        "right_fingers": "purple",
        "left_fingers": "magenta",
        "connector": "gray",
    }

    # Updated connection groups
    POSE22_CONNECTIONS_COLORED = {
        "right_leg": [(0, 1), (1, 2), (2, 3)],                  # RKnee → RAnkle → RToe → RHeel
        "connector": [(0, 8), (4, 8)],                          # Knees → Spine base
        "left_leg": [(4, 5), (5, 6), (6, 7)],                   # LKnee → LAnkle → LToe → LHeel
        "spine": [(8, 9), (9, 10), (10, 11)],                   # Spine → Thorax → Neck → Head
        "left_arm": [(9, 12), (12, 13), (13, 14)],              # Thorax → LShoulder → LElbow → LWrist
        "right_arm": [(9, 17), (17, 18), (18, 19)],             # Thorax → RShoulder → RElbow → RWrist
        "right_fingers": [(19, 20), (20, 21)],                  # RWrist → RFinger1 → RFinger2
        "left_fingers": [(14, 15), (15, 16)],                   # LWrist → LFinger1 → LFinger2
    }

    x = joints_2d[:, 0]
    y = joints_2d[:, 1]

    for part, connections in POSE22_CONNECTIONS_COLORED.items():
        for start, end in connections:
            if start < len(x) and end < len(x):
                plt.plot([x[start], x[end]], [y[start], y[end]], color=color_map[part], linewidth=2)

    plt.scatter(x, y, color='red', s=10)
    # plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.pause(0.01)

def animate_2d_pose_sequence(pose_sequence, pause_time=0.2):
    for i, joints in enumerate(pose_sequence):
        plt.clf()

        # Define joint region color mapping
        color_map = {
            "right_leg": "deepskyblue",
            "left_leg": "royalblue",
            "spine": "black",
            "head": "darkorange",
            "left_arm": "forestgreen",
            "right_arm": "seagreen",
            "right_fingers": "purple",
            "left_fingers": "magenta",
            "connector": "gray",
        }

        # Updated connection groups
        POSE22_CONNECTIONS_COLORED = {
            "right_leg": [(0, 1), (1, 2), (2, 3)],                  # RKnee → RAnkle → RToe → RHeel
            "connector": [(0, 8), (4, 8)],                          # Knees → Spine base
            "left_leg": [(4, 5), (5, 6), (6, 7)],                   # LKnee → LAnkle → LToe → LHeel
            "spine": [(8, 9), (9, 10), (10, 11)],                   # Spine → Thorax → Neck → Head
            "left_arm": [(9, 12), (12, 13), (13, 14)],              # Thorax → LShoulder → LElbow → LWrist
            "right_arm": [(9, 17), (17, 18), (18, 19)],             # Thorax → RShoulder → RElbow → RWrist
            "right_fingers": [(19, 20), (20, 21)],                  # RWrist → RFinger1 → RFinger2
            "left_fingers": [(14, 15), (15, 16)],                   # LWrist → LFinger1 → LFinger2
        }

        # Compute global limits once
        all_x = pose_sequence[:, :, 0].flatten()
        all_y = pose_sequence[:, :, 1].flatten()
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        padding = 10  # Add some space around for visibility

        for group, connections in POSE22_CONNECTIONS_COLORED.items():
            color = color_map[group]
            for start, end in connections:
                if start < joints.shape[0] and end < joints.shape[0]:
                    x = [joints[start, 0], joints[end, 0]]
                    y = [joints[start, 1], joints[end, 1]]
                    plt.plot(x, y, color=color, linewidth=2)

        plt.scatter(joints[:, 0], joints[:, 1], color='red', s=10)
        # plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlim(x_min - padding, x_max + padding)
        plt.ylim(y_min - padding, y_max + padding)
        plt.title(f"Live Predicted Pose (Frame {i+1})")
        plt.pause(pause_time)
# Chatted

def process_data(filename, num_frames=50):
    info = open(filename, 'r').readlines()
    if len(info) < num_frames:
        return None

    lines = info[-num_frames:] # take last num_frames 
    pose_info = [] 

    # remove commas, append to pose_info
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            pose_info.append(np.array([float(x) for x in line]))
            
    pose_info = np.array(pose_info) # turn into np array, [x, num_ppl * 75]
    pose_info = pose_info[:, 0:75] # take first human
    pose_info = pose_info.reshape(-1, 25, 3) # into [x, num_joints, 3]

    pose_32 = np.zeros((pose_info.shape[0], 33, 3), dtype=np.float32) # get zeros for h36 format
    pose_32[:, pose32_target_indices, :] = pose_info[:, body25_indices, :] # set corresponding 

    pose_32[:, :2] = 0 # set first two joints to zero?? 

    N = pose_32.shape[0] 

    pose_info = pose_32.reshape(-1, 3) # reshape into (x*33, 3)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    # new stuff
    print("NEW STUFFFF -------------------")

    num_frames = 25
    predicted_frames = 10
    
    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    input_filename = '/home/eric/Downloads/DeepRob/simlpe_repo/deeprob_siMLPe/data/test.txt'

    prev_num_lines = 0

    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(5, 5))  # Create once, reuse

    while(True):

        with open(input_filename, 'r') as f:
            lines = f.readlines()

        if len(lines) == prev_num_lines:
            continue

        prev_num_lines = len(lines)

        live_data = process_data(input_filename, num_frames)

        if live_data is None or live_data.shape[0] < 25:
            continue

        data = live_data[:, joint_used_xyz, :].reshape(live_data.shape[0], -1).unsqueeze(0)
        data = data.cuda()

        with torch.no_grad():
            prediction = model(data)[:, :predicted_frames]
            pred_xyz = prediction.reshape(predicted_frames, 22, 3)

        print('prediction output', prediction.shape)

        latest_frame_2d = pred_xyz[-1][:, :2].cpu().numpy()
        # show_pose_2d(latest_frame_2d, title="Live Predicted Pose")
        joints_2d_seq = pred_xyz[:, :, :2].cpu().numpy()  # Just X, Y

        animate_2d_pose_sequence(joints_2d_seq)

        output_filename = "/home/eric/Downloads/DeepRob/simlpe_repo/deeprob_siMLPe/exps/baseline_h36m/outputs.txt"

        # Save each frame as a line: x1,y1,z1,x2,y2,z2,...
        with open(output_filename, 'w') as f:
            for frame in pred_xyz.cpu().numpy():
                flat = frame.flatten()  # shape (66,)
                line = ','.join([f"{x:.6f}" for x in flat])
                f.write(line + '\n')


        # write data to output text file


