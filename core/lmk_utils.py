import cv2
import numpy as np
from PIL import Image
import torch
import random

def plot_kpt(image, st, ed, color, thickness=1):
    image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), color, thickness, cv2.LINE_AA)
    return image


def set_random_colors(num=68+3):
    colors = []
    cube_root = num ** (1.0 / 3)
    for r in range(0, 255, int(255 / cube_root)):
        for g in range(0, 255, int(255 / cube_root)):
            for b in range(0, 255, int(255 / cube_root)):
                colors.append([int(r), int(g), int(b)])
    random.shuffle(colors)
    return colors

def get_nearest_point_index(lmks_points, img_size=1024, div_amount = 10):
    if lmks_points.shape[-1] == 3:
        lmks_points = lmks_points[:,:-1]
        
    intervals = np.linspace(0, img_size-1, div_amount, dtype=int)[:,np.newaxis]
    l = np.concatenate([np.array([[0]]*intervals.shape[0]), intervals], axis=-1) # (x, y)
    d = np.concatenate([intervals, np.array([[img_size-1]]*intervals.shape[0]),], axis=-1)
    r = np.concatenate([np.array([[img_size-1]]*intervals.shape[0]), intervals], axis=-1)[::-1,...]
    u = np.concatenate([intervals, np.array([[0]]*intervals.shape[0])], axis=-1)[::-1,...]

    edge_points = np.concatenate([l,d,r,u],axis=0)
    nearest_point_indexes = []
    for edge_point in edge_points:
        _edge_point = edge_point[np.newaxis,:].repeat(len(lmks_points),0)
        # lmks_points
        distance = np.linalg.norm(_edge_point - lmks_points, axis=-1)
        min_index = np.where(distance==distance.min())[0][0]
        nearest_point_indexes.append(min_index)
        
    nearest_points = lmks_points[nearest_point_indexes]
    return edge_points, nearest_points, nearest_point_indexes

def get_contour_vis(canvas, edge_points, nearest_points, colors, dense = 5):
    # main lines
    inter_points = np.array([])
    for idx, (edge_point, nearest_point) in enumerate(zip(edge_points, nearest_points)):
        
        canvas = plot_kpt(canvas, edge_point, nearest_point, colors[idx])
        
        
        # intermediate points
        inter_points = np.append(inter_points, np.linspace(edge_point, nearest_point, dense+2, dtype=int)[1:-1])
        
    inter_groups = np.reshape(inter_points, (-1, dense, 2))
    _inter_groups = np.append(inter_groups, inter_groups[0][np.newaxis,...],axis=0).swapaxes(0,1)
    _inter_groups = np.stack([_inter_groups[1],_inter_groups[3],_inter_groups[4]])
        # roll_inter_group = np.roll(inter_group, shift=1, axis=0)
    
    for inter_group in _inter_groups:
        st_pts, ed_pts = inter_group[:-1], inter_group[1:]
        # _inter_group = 
        for _idx, (st_pt, ed_pt) in enumerate(zip(st_pts, ed_pts),start=idx):
            canvas = plot_kpt(canvas, st_pt, ed_pt, colors[_idx])
        idx = _idx
        
    return canvas


colors = set_random_colors(10*4*5 + 9*4)

lmks = np.load('/data1/KTH-dataset/KTH_dataset_align_warp_1024/18/hanbam2_000382_3_4/lmks_756_np/003926.npy')
vis = cv2.imread('/data1/KTH-dataset/KTH_dataset_align_warp_1024/18/hanbam2_000382_3_4/lmks_756_vis/003926.png')



edge_points, nearest_points, nearest_point_indexes = get_nearest_point_index(lmks)
canvas = get_contour_vis(vis, edge_points, nearest_points, colors)
cv2.imwrite('./canvas.png', canvas)