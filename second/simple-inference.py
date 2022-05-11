# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import numpy as np
import pickle
from pathlib import Path
import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
import datetime
import random
import matplotlib.pyplot as plt
config_path = "/home/reinht/Desktop/second.pytorch/second/configs/pointpillars/car/xyres_16.config"
config = pipeline_pb2.TrainEvalPipelineConfig()
with open(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, config)
input_cfg = config.eval_input_reader
model_cfg = config.model.second
#config_tool.change_detection_range_v2(model_cfg, [-50, -50, 50, 50])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
ckpt_path = "/home/reinht/pointpillars/voxelnet-296960.tckpt"
net = build_network(model_cfg).to(device).eval()
net.load_state_dict(torch.load(ckpt_path))
target_assigner = net.target_assigner
voxel_generator = net.voxel_generator
grid_size = voxel_generator.grid_size
feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
feature_map_size = [*feature_map_size, 1][::-1]
anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
anchors = anchors.view(1, -1, 7)
info_path = input_cfg.dataset.kitti_info_path
root_path = Path(input_cfg.dataset.kitti_root_path)
with open(info_path, 'rb') as f:
    infos = pickle.load(f)
time_list = []
print(len(infos))
for i in range(1):
    begin_time = datetime.datetime.now()
    index = random.randint(0,3768)
    info = infos[index]
    v_path = info["point_cloud"]['velodyne_path']
    v_path = str(root_path / v_path)
    print(v_path)
    points = np.fromfile(
        v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    res = {}
    res = voxel_generator.generate(points, max_voxels=90000)
    voxels = res['voxels']
    coordinates = res['coordinates']
    num_points_per_voxel = res['num_points_per_voxel']
    #print(voxels.shape)
    # add batch idx to coords
    coords = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points_per_voxel, dtype=torch.int32, device=device)
    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
    }
    pred = net(example)[0]
    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
    plt.imshow(bev_map)
    # deltatime = datetime.datetime.now() - begin_time
    # time_list.append(deltatime)
# average = sum(time_list, datetime.timedelta())/len(time_list)
# max = max(time_list)
# min = min(time_list)
# print("Average time: ", average)
# print("Max time: ", max)
# print("Min time: ", min)
#print(pred)
