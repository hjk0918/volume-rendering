# from render_volume_misc import gkern_2d, gkern_3d, obb2hbb, density_to_alpha, world2grid, 
# from scipy.ndimage.filters import gaussian_filter
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
from pyvista import examples
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import cv2
import os
from os.path import join
from argparse import ArgumentParser
import json
import copy
from scipy.ndimage import gaussian_filter
from bbox_proj import project_obb_to_image
import shutil
from copy import deepcopy
import subprocess
import glob

def gkern_3d(w=10, l=10, h=3, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    Reference: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    ay = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    az = np.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    gauss_x = np.exp(-0.5 * np.square(ax) / np.square(w/5))
    gauss_y = np.exp(-0.5 * np.square(ay) / np.square(l/5))
    gauss_z = np.exp(-0.5 * np.square(az) / np.square(h/5))
    kernel = np.outer(np.outer(gauss_x, gauss_y), gauss_z).reshape(w, l, h)
    return kernel

def obb2point8(obboxes):
    """
    Args:
        obboxes (N, 7): [x, y, z, w, l, h, theta]
    Returns:
        obboxes_8 (N, 8, 3): 8 corners of the obboxes
    """
    x, y, z, w, l, h, theta = np.split(obboxes, [1, 2, 3, 4, 5, 6], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias1 = w/2 * Cos - l/2 * Sin
    x_bias2 = w/2 * Cos + l/2 * Sin
    y_bias1 = w/2 * Sin + l/2 * Cos
    y_bias2 = w/2 * Sin - l/2 * Cos
    xy1 = np.concatenate([x+x_bias1, y+y_bias1], axis=-1)
    xy2 = np.concatenate([x+x_bias2, y+y_bias2], axis=-1)
    xy3 = np.concatenate([x-x_bias1, y-y_bias1], axis=-1)
    xy4 = np.concatenate([x-x_bias2, y-y_bias2], axis=-1)
    z1, z2 = z-h/2, z+h/2
    return np.concatenate([xy1, z1, xy2, z1, xy3, z1, xy4, z1, 
                           xy1, z2, xy2, z2, xy3, z2, xy4, z2,], axis=-1).reshape(-1, 8, 3)

def aabb2point8(aabbs):
    """
    Args:
        aabbs (N, 6): [x1, y1, z1, x2, y2, z2]
    Returns:
        aabbs_8 (N, 8, 3): 8 corners of the aabbs
    """
    x1, y1, z1, x2, y2, z2 = np.split(aabbs, [1, 2, 3, 4, 5], axis=-1)
    return np.concatenate([x1, y1, z1, x2, y1, z1, x2, y2, z1, x1, y2, z1, 
                           x1, y1, z2, x2, y1, z2, x2, y2, z2, x1, y2, z2], axis=-1).reshape(-1, 8, 3).astype(np.float32)


def obb2hbb(obboxes):
    """Return the smallest 3D AABB that contains the 3D OBB."""
    center, z, w, l, h, theta = np.split(obboxes, [2, 3, 4, 5, 6], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = (np.abs(w/2 * Cos) + np.abs(l/2 * Sin))
    y_bias = (np.abs(w/2 * Sin) + np.abs(l/2 * Cos))
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return np.concatenate([center-bias, z-h/2, center+bias, z+h/2], axis=-1)

def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

def world2grid(points, room_bbox, res, downsample=1):
    # points (..., 3)
    # room_bbox [xmin, ymin, zmin, xmax, ymax, zmax]
    points -= room_bbox[:3]
    points /= np.max(room_bbox[3:] - room_bbox[:3])
    points *= np.max(res)
    return points / downsample

def grid2world(points, room_bbox, res):
    points /= np.max(res)
    points *= np.max(room_bbox[3:] - room_bbox[:3])
    points += room_bbox[:3]
    return points

def ngp_matrix_to_nerf(ngp_matrix, scale, offset, from_mitsuba):
    result = deepcopy(ngp_matrix)
    if from_mitsuba:
        result[:, [0, 2]] *= -1
    else:
        # Cycle axes xyz->yzx
        result = result[[2, 0, 1], :]
    
    result[:, [1, 2]] *= -1
    result[:, 3] = (result[:, 3] - offset) / scale
    return result

def ngp_aabb_to_nerf(ngp_aabb, scale, offset, from_mitsuba):
    offset = np.array(offset)
    result = deepcopy(ngp_aabb)
    if from_mitsuba:
        raise KeyError("Not implemented")
        result[:, [0, 2]] *= -1
    else:
        # Cycle axes xyz->yzx
        result = result[:, [2, 0, 1]]
    
    # result[:, [1, 2]] *= -1
    result = (result - offset) / scale
    return result

def load_alpha_and_proposals(feature_path: str, proposal_path: str, json_path: str, args):
    """ Load alpha and proposals from the given paths. 
    Args:
        feature_path (str): path to the alpha feature
        proposal_path (str): path to the proposals
        json_path (str): path to the json file
        transpose_yz (bool): whether to transpose the y and z axis
    Returns:
        alpha (np.ndarray): alpha feature
        proposals (np.ndarray, Nx6): aabb proposals
        room_bbox (np.ndarray, 6): room bounding box
        res (np.ndarray, 3): resolution of the alpha feature
        boxes_8: (np.ndarray, Nx8x3): 8 corners of the obb proposals in the world coordinate
    """
    feature_npz = np.load(feature_path)
    rgbsigma = feature_npz['rgbsigma']
    res = feature_npz['resolution']
    scale = feature_npz['scale']
    offset = feature_npz['offset']
    from_mitsuba = feature_npz['from_mitsuba']
    bbox_min = feature_npz['bbox_min']
    bbox_max = feature_npz['bbox_max']

    with open(json_path, 'r') as f:
        json_dict = json.load(f)
        if 'room_bbox' in json_dict:
            room_bbox = np.array(json_dict['room_bbox']).flatten()
        else:
            room_bbox = ngp_aabb_to_nerf(np.array([bbox_min, bbox_max]), scale, offset, from_mitsuba).flatten()
    if args.dataset == 'scannet':
        room_bbox = room_bbox[[1, 2, 0, 4, 5, 3]]
    
    # First reshape from (H * W * D, C) to (D, H, W, C)
    rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1)
    if args.dataset == 'scannet':
        alpha = rgbsigma[..., -1]
    else:
        alpha = density_to_alpha(rgbsigma[..., -1])

    if args.transpose_yz:
        # Transpose to (D, W, H)
        alpha = np.transpose(alpha, (0, 2, 1))
        res = [res[2], res[0], res[1]]
    else:
        # Transpose to (W, H, D)
        alpha = np.transpose(alpha, (2, 1, 0))
        # res = [res[1], res[2], res[0]]
    

    proposals_npz = np.load(proposal_path)
    if 'proposals' in proposals_npz:
        proposals = proposals_npz['proposals']
    elif 'proposal' in proposals_npz:
        proposals = proposals_npz['proposal']
    else:
        raise ValueError('proposals and proposal are not found in npz.')
    
    return alpha, proposals, room_bbox, res

def visualize_3dgrid(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    grid = np.concatenate([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1), 
                           heatmap.reshape(-1, 1)], axis=-1)
    grid = np.concatenate([grid, np.ones((grid.shape[0], 1))], axis=-1)
    df = pd.DataFrame(grid, columns=['x', 'y', 'z', 'value', 'species'])
    import plotly.express as px
    fig = px.scatter_3d(df, x='x', y='y', z='z', size='value', color='value', symbol='species')
    fig.show()

def visualize_3dgrid_go(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    x, y, z, heatmap_f = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), heatmap.flatten()
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=heatmap_f*50,
            color=heatmap_f,                # set color to an array/list of desired values
            colorscale='Hot',   # choose a colorscale
            opacity=0.8,
            symbol='circle'
        )
    )])
    fig.show()

def visualize_volume(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    x, y, z, heatmap_f = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), heatmap.flatten()
    fig = go.Figure(data=go.Volume(
        x=x,
        y=y,
        z=z,
        value=heatmap_f,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=100, # needs to be a large number for good volume rendering
        ))
    fig.show()

def normalize_heatmap(heatmap, sigma=0):
    if sigma>0:
        heatmap = gaussian_filter(heatmap, sigma)
    mean, std = heatmap.mean(), heatmap.std()
    heatmap = (heatmap - mean) / std
    return heatmap

from scipy.ndimage import zoom
def generate_heatmap(alpha, boxes, scene_name, args):

    voxel_hmp = np.zeros_like(alpha)
    voxel_npz = np.load(join(args.voxel_dir, scene_name + '.npz'))
    for key in voxel_npz.files:
        feature = voxel_npz[key]
        scale = np.array(voxel_hmp.shape).astype(float) / np.array(feature.shape).astype(float)
        voxel_hmp += zoom(feature,scale,mode='nearest')
    
    box_hmp = np.zeros_like(alpha)
    # for box in boxes:
    #     kernel = np.zeros((box[3]-box[0], box[4]-box[1], box[5]-box[2]))
    #     if args.kernel_type == 'gaussian':
    #         kernel = gkern_3d(w=box[3]-box[0], l=box[4]-box[1], h=box[5]-box[2])
    #     elif args.kernel_type == 'box':
    #         kernel = np.ones_like(kernel)
    #     box_hmp[box[0]:box[3], box[1]:box[4], box[2]:box[5]] += kernel
    
    if args.hmp_type == 'voxel':
        heatmap = voxel_hmp
    elif args.hmp_type in ['proposal', 'gt']:
        heatmap = box_hmp
    elif args.hmp_type == 'mix':
        heatmap = 0.5 * normalize_heatmap(voxel_hmp) + 0.5 * normalize_heatmap(box_hmp)
    
    heatmap = normalize_heatmap(heatmap, args.gaussian_sigma)
    return heatmap

def get_overview_position(room_bbox, res, downsample, in_offset=0.3, cam_height=2, index=2):
    x1, y1, x2, y2 = room_bbox[0]+in_offset, room_bbox[1]+in_offset, room_bbox[3]-in_offset, room_bbox[4]-in_offset
    focal_point = np.array([(x1+x2)/2., (y1+y2)/2., 1.0])
    cam_positions = np.array([[x1, y1, cam_height], [x1, y2, cam_height], [x2, y1, cam_height], [x2, y2, cam_height]])
    focal_point = world2grid(focal_point, room_bbox, res, downsample)
    cam_positions = world2grid(cam_positions, room_bbox, res, downsample)

    return focal_point, cam_positions[index]

def merge_images(img1, img2, alpha=0.5):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    img = alpha * img1 + (1-alpha) * img2
    return img.astype(np.uint8)

transpose_yzx = np.array([[0, 1, 0, 0], 
                          [0, 0, 1, 0], 
                          [1, 0, 0, 0], 
                          [0, 0, 0, 1]])
transpose_zxy = np.array([[0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]])

def frame2config(frame_list, room_bbox, res, args):
    """ Decode frame list to get camera positions and focal points in grid. 
    """
    names, cam_positions, focal_points, poses = [], [], [], []
    for frame_meta in frame_list:
        name = frame_meta['file_path'].split('/')[-1][:-4]
        names.append(name)

        c2w = np.array(frame_meta['transform_matrix'])
        poses.append(c2w)

        cam_position_world = copy.deepcopy(c2w[:3, 3])
        cam_position = world2grid(cam_position_world, room_bbox, res, args.downsample)
        cam_positions.append(cam_position)

        focal_point_hom = c2w @ np.array([0, 0, -1, 1]).T # flip z axis
        focal_point_world = focal_point_hom[:3] / focal_point_hom[3]
        focal_point = world2grid(focal_point_world, room_bbox, res, args.downsample)
        focal_points.append(focal_point)

    
    return names, cam_positions, focal_points, poses

def render_volume(heatmap, room_bbox, res, output_dir, args, json_dict=None, boxes_8=None, frame_list=None):
    heatmap = heatmap[0::args.downsample, 0::args.downsample, 0::args.downsample]
    heatmap *= args.value_scale 
    # print('mean={}, std={}, min={}, max={}'.format(np.mean(heatmap), np.std(heatmap), np.min(heatmap), np.max(heatmap)))

    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(heatmap.shape) + 1

    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = heatmap.flatten(order="F")  # Flatten the array!
    # grid.cell_data["alpha"] = alpha.flatten(order="F")  

    # Now plot the grid!
    # grid.plot(show_edges=True)
    # print(grid)

    p = pv.Plotter()
    opacity = [1, 1, 1, 1, 1.0]
    p.add_volume(heatmap, cmap='jet', opacity=opacity, blending='maximum')
    # p.add_volume(grid, cmap='jet', opacity=opacity, blending='maximum')
    p.background_color = "black"
    # p.add_background_image('/Users/abraham/Desktop/fyp/RPN_NeRF_temp/raw_2.jpg')
    p.window_size = [640, 480]
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = args.view_angle
    p.remove_scalar_bar()
    
    if frame_list!=None: # debug
        # focal_point, cam_position = get_overview_position(room_bbox, res, args.downsample, index=2)
        cam_position = frame_list[0]['cam_position']
        focal_point = frame_list[0]['focal_point']
        p.camera.position = cam_position
        p.camera.focal_point = focal_point
        p.show()
        # p.save_graphic(join(output_dir, "temp/heatmap.svg"))
    else:
        frame_dict = json_dict['frames']
        names, cam_positions, focal_points, poses = frame2config(frame_dict, room_bbox, res, args)
        if args.dataset == 'scannet':
            fl_x, fl_y, cx, cy = frame_dict[0]['fx'], frame_dict[0]['fy'], frame_dict[0]['cx'], frame_dict[0]['cy']
        else:
            fl_x, fl_y, cx, cy = json_dict['fl_x'], json_dict['fl_y'], json_dict['cx'], json_dict['cy']
        intrinsic_mat = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
        
        for name, cam_position, focal_point, pose in zip(names, cam_positions, focal_points, poses):
            p.camera.position = cam_position
            p.camera.focal_point = focal_point
            if args.interactive:
                p.show()
                break
            else:
                if os.path.isfile(join(args.dataset_dir, args.scene_name, 'val/screenshots/'+name+'.jpg')):
                    input = cv2.imread(join(args.dataset_dir, args.scene_name, 'val/screenshots/'+name+'.jpg'))
                elif os.path.isfile(join(args.dataset_dir, args.scene_name, 'val/screenshots/'+name+'.png')):
                    input = cv2.imread(join(args.dataset_dir, args.scene_name, 'val/screenshots/'+name+'.png'))
                elif os.path.isfile(join(args.dataset_dir, args.scene_name, 'render/'+name+'.jpg')):
                    input = cv2.imread(join(args.dataset_dir, args.scene_name, 'render/'+name+'.jpg'))
                else:
                    continue
                    # raise ValueError('No input image found for {}'.format(name))
                p.save_graphic(join(output_dir, scene_name+'_'+name+'_hmp.svg'))
                renderPM.drawToFile(svg2rlg(join(output_dir, scene_name+'_'+name+'_hmp.svg')), 
                                            join(output_dir, scene_name+'_'+name+'_hmp.png'), fmt='PNG')
                os.remove(join(output_dir, scene_name+'_'+name+'_hmp.svg'))
                hmp = cv2.imread(join(output_dir, scene_name+'_'+name+'_hmp.png'))
                
                shape = input.shape
                hmp = cv2.resize(hmp, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
                alpha, beta, gamma = args.blend_alpha_beta_gamma
                blended = cv2.addWeighted(input, alpha, hmp, beta, gamma)
                output = copy.deepcopy(input)
                output = project_obb_to_image(output, intrinsic_mat, np.linalg.inv(pose), boxes_8, line_width=args.line_width)

                os.makedirs(join(output_dir, 'split'), exist_ok=True)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_input.png'), input)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_hmp.png'), hmp)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_blend.png'), blended)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_output.png'), output)

                concat1 = np.concatenate([input, hmp], axis=1)
                concat2 = np.concatenate([blended, output], axis=1)
                concat = np.concatenate([concat1, concat2], axis=0)
                cv2.imwrite(join(output_dir, scene_name+'_'+name+'_hmp.png'), concat)

def render_video(imgs, heatmap, room_bbox, res, output_dir, args, val_json_dict, traj_json_dict, boxes_8=None, frame_list=None):
    heatmap = heatmap[0::args.downsample, 0::args.downsample, 0::args.downsample]
    heatmap *= args.value_scale 
    # print('mean={}, std={}, min={}, max={}'.format(np.mean(heatmap), np.std(heatmap), np.min(heatmap), np.max(heatmap)))

    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(heatmap.shape) + 1

    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = heatmap.flatten(order="F")  # Flatten the array!
    # grid.cell_data["alpha"] = alpha.flatten(order="F")  

    # Now plot the grid!
    # grid.plot(show_edges=True)
    # print(grid)

    p = pv.Plotter()
    opacity = [1, 1, 1, 1, 1.0]
    p.add_volume(heatmap, cmap='jet', opacity=opacity, blending='maximum')
    # p.add_volume(grid, cmap='jet', opacity=opacity, blending='maximum')
    p.background_color = "black"
    # p.add_background_image('/Users/abraham/Desktop/fyp/RPN_NeRF_temp/raw_2.jpg')
    p.window_size = [1920, 1080]
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = args.view_angle
    p.remove_scalar_bar()
    
    if frame_list!=None: # debug
        # focal_point, cam_position = get_overview_position(room_bbox, res, args.downsample, index=2)
        cam_position = frame_list[0]['cam_position']
        focal_point = frame_list[0]['focal_point']
        p.camera.position = cam_position
        p.camera.focal_point = focal_point
        p.show()
        # p.save_graphic(join(output_dir, "temp/heatmap.svg"))
    else:

        frame_dict = traj_json_dict['frames']
        names, cam_positions, focal_points, poses = frame2config(frame_dict, room_bbox, res, args)
        if args.dataset == 'scannet':
            fl_x, fl_y, cx, cy = frame_dict[0]['fx'], frame_dict[0]['fy'], frame_dict[0]['cx'], frame_dict[0]['cy']
        else:
            angle_x = traj_json_dict["camera_angle_x"]
            cx, cy = traj_json_dict["w"]/2, traj_json_dict["h"]/2
            fl_x = cx / np.tan(angle_x / 2)
            fl_y = fl_x
        intrinsic_mat = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

        # intrinsic_mat = np.array([[1150, 0, 960], [0, 1150, 540], [0, 0, 1]])
        
        for idx, (name, cam_position, focal_point, pose, input) in enumerate(zip(names, cam_positions, focal_points, poses, imgs)):
            # if idx not in [15, 102, 103, 104, 112, 119, 120, 121, 167, 168, 171, 172, 173, 187, 
            #                 222, 230, 249, 254, 293, 297, 298, 299, 300]:
            #     continue

            # if idx not in [168]:
            #     continue
            
            p.camera.position = cam_position
            p.camera.focal_point = focal_point
            if args.interactive:
                p.show()
                break
            else:
                p.save_graphic(join(output_dir, scene_name+'_'+name+'_hmp.svg'))
                renderPM.drawToFile(svg2rlg(join(output_dir, scene_name+'_'+name+'_hmp.svg')), 
                                            join(output_dir, scene_name+'_'+name+'_hmp.png'), fmt='PNG')
                os.remove(join(output_dir, scene_name+'_'+name+'_hmp.svg'))
                hmp = cv2.imread(join(output_dir, scene_name+'_'+name+'_hmp.png'))
                os.remove(join(output_dir, scene_name+'_'+name+'_hmp.png'))
                
                shape = input.shape
                hmp = cv2.resize(hmp, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
                alpha, beta, gamma = args.blend_alpha_beta_gamma
                blended = cv2.addWeighted(input, alpha, hmp, beta, gamma)
                output = copy.deepcopy(input)
                output = project_obb_to_image(output, intrinsic_mat, np.linalg.inv(pose), boxes_8, line_width=args.line_width)

                os.makedirs(join(output_dir, 'split'), exist_ok=True)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_hmp.png'), hmp)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_blend.png'), blended)
                cv2.imwrite(join(output_dir, 'split', scene_name+'_'+name+'_output.png'), output)

                # if (idx+1) % 30 == 0:
                #     break

                # if True:
                if idx % 50 == 0:
                    concat1 = np.concatenate([input, hmp], axis=1)
                    concat2 = np.concatenate([blended, output], axis=1)
                    concat = np.concatenate([concat1, concat2], axis=0)
                    cv2.imwrite(join(output_dir, scene_name+'_'+name+'_hmp.png'), concat)
            
        
        kwlist = ['hmp', 'blend', 'output']
        for kw in kwlist:
            img_path_list = sorted(glob.glob(join(output_dir, 'split', f'*_{kw}.png')))
            print(join(output_dir, scene_name, f'video_{kw}.mp4'))
            kargs = {'macro_block_size': None}
            writer = imageio.get_writer(join(output_dir, f'video_{kw}.mp4'), fps=30, **kargs)
            for im in img_path_list:
                writer.append_data(imageio.v2.imread(im))
            writer.close()

def merge_videos(video1_path, video2_path, img_path):
    bev_img = cv2.imread(img_path)
    vid1 = imageio.get_reader(video1_path,  'ffmpeg')
    vid2 = imageio.get_reader(video2_path,  'ffmpeg')
    vid1_num_frames = vid1.count_frames()
    vid2_num_frames = vid2.count_frames()
    assert vid1_num_frames == vid2_num_frames, "The number of frames in two videos should be the same."
    vid_imgs = []
    for num in range(vid2_num_frames):
        img1 = cv2.cvtColor(vid1.get_data(num), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(vid2.get_data(num), cv2.COLOR_RGB2BGR)
        print(img1.dtype)
        print(img2.dtype)
        print(bev_img.dtype)

        h, w, _ = img1.shape
        re_h = h * 2
        re_w = re_h * 4 / 3

        img = np.zeros((re_h, re_w, 3), dtype=np.uint8)
        img[0:h, -w:, :] = img1
        img[h:, -w:, :] = img2
        


        imgs.append(img)


def select_and_blend():
    pass

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument('--feature_dir', type=str, help='path to feature directory')
    parser.add_argument('--target_dir', type=str, help='path to target directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--boxes_dir', type=str, help='path to boxes directory')
    parser.add_argument('--transpose_yz', action='store_true', help='transpose y and z')
    parser.add_argument('--hmp_top_k', type=int, default=100, help='top k proposals to be used for heatmap.')
    parser.add_argument('--vis_top_n', type=int, default=-1, help='top n proposals to be used for projection visualization.')
    parser.add_argument('--hmp_type', type=str, default='voxel', choices=['voxel', 'proposal', 'mix', 'gt'], 
                        help='type of heatmap')
    parser.add_argument('--kernel_type', type=str, default='gaussian', choices=['gaussian', 'box'], 
                        help='type of heatmap to be generated')
    parser.add_argument('--value_scale', type=float, default=20, help='value scaling for heatmap')
    parser.add_argument('--downsample', type=int, default=2, help='downsample factor for heatmap')
    parser.add_argument('--gaussian_sigma', type=float, default=5, help='sigma for gaussian kernel')
    parser.add_argument('--concat_img', action='store_true', help='concatenate heatmap with NeRF image')
    parser.add_argument('--interactive', action='store_true', help='interactive mode, used in .ipynb')
    parser.add_argument('--blend_alpha_beta_gamma', type=float, nargs=3, default=[0.7, 0.6, 20], 
                        help='blended_img = alpha*input + beta*heatmap + gamma')
    parser.add_argument('--line_width', type=int, default=4, help='line width for proposal visualization')
    parser.add_argument('--command_path', type=str, default='', help='path to sh command for archiving')
    parser.add_argument('--single_scene', type=str, default='', help='only process one designated scene')
    parser.add_argument('--view_angle', type=int, default=65, help='view angle for projection visualization')
    parser.add_argument('--dataset', type=str, help='dataset name')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    args.proposal_dir = join(args.target_dir, 'proposals')
    args.voxel_dir = join(args.target_dir, 'voxel_scores')

    scene_list = [x.split('.')[0] for x in sorted(os.listdir(args.voxel_dir))]
    if args.single_scene != '':
        scene_list = [args.single_scene]
    # debug
    # scene_list = ['ai_016_010']
    # scene_list = ['scene0487_00']
    # scene_list = ['scene0002_00', 'scene0166_00', 'scene0220_00', 'scene0416_00', 'scene0487_00']
    # scene_list = scene_list[:1]
    # scene_list = ['025', '054', '078', '225', '276', '522']
    # scene_list = ['054']
    # scene_list = ['3dfront_0037_00']
    # scene_list = ['3dfront_0091_00', '3dfront_0072_00', '3dfront_0037_00']

    for scene_name in scene_list:
        if scene_name == 'scene0040_00':
            continue
        args.scene_name = scene_name # pass scene_name to with args
        feature_path = join(args.feature_dir, scene_name+'.npz')
        proposal_path = join(args.proposal_dir, scene_name+'.npz')
        if args.dataset == 'scannet':
            val_json_path = join(args.dataset_dir, scene_name, 'transforms_test.json')
        else:
            val_json_path = join(args.dataset_dir, scene_name, 'val', 'val_transforms.json')
        assert os.path.isfile(feature_path), 'feature file not found: {}'.format(feature_path)
        # assert os.path.isfile(proposal_path), 'proposal file not found: {}'.format(proposal_path)
        scene_output_dir = join(args.output_dir, scene_name)
        # if os.path.isdir(scene_output_dir):
        #     shutil.rmtree(scene_output_dir)
        os.makedirs(scene_output_dir, exist_ok=True)

        alpha, proposals, room_bbox, res = load_alpha_and_proposals(feature_path, proposal_path, val_json_path, args)
        aabbs = obb2hbb(proposals[:args.hmp_top_k]).astype(int)
        boxes_point8 = grid2world(obb2point8(proposals[:args.vis_top_n]), room_bbox, res)
        # boxes_point8 = grid2world(obb2point8(proposals[2:4]), room_bbox, res)
        
        os.environ['IMAGEIO_FFMPEG_EXE'] = '/Users/abraham/Desktop/fyp/RPN_NeRF_temp/ffmpeg'
        import imageio
        video_path = join('./supmat_resources', scene_name, 'video.mp4')
        vid = imageio.get_reader(video_path,  'ffmpeg')
        num_frames=vid.count_frames()
        imgs = []
        for num in range(num_frames):
            image = vid.get_data(num)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imgs.append(image)
        
        with open(join('./supmat_resources', scene_name, 'ref_xform.json'), 'r') as f:
            traj_json_dict = json.load(f)
        print(len(imgs))
        print(len(traj_json_dict['frames']))

        heatmap = generate_heatmap(alpha, aabbs, scene_name, args)

        np.set_printoptions(precision=2, suppress=True)
        print(heatmap.shape)
        print(res)
        print(room_bbox)
        # exit()

        # render poses in val_json_paths
        with open(val_json_path, 'r') as f:
            val_json_dict = json.load(f)
        
        # render_video(imgs, heatmap, room_bbox, res, scene_output_dir, args, val_json_dict, traj_json_dict, boxes_point8)
        
        merge_videos(join(scene_output_dir, 'video_output.mp4'), 
                    join(scene_output_dir, 'video_blend.mp4'), 
                    join('./supmat_resources', scene_name, 'bev.png'))


        subprocess.run(['cp', args.command_path, join(args.output_dir, scene_name)])

    select_and_blend()


        

    print('Done.')