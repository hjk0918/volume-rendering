from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def get_aabb_coords(aabb_code):
    """Return a list of homogeneous 3D coords of the bbox. """
    [x1, y1, z1, x2, y2, z2] = aabb_code
    bbox_coords = np.array([[x1, y1, z1],
                           [x1, y2, z1],
                           [x2, y2, z1],
                           [x2, y1, z1],
                           [x1, y1, z2],
                           [x1, y2, z2],
                           [x2, y2, z2],
                           [x2, y1, z2]])
    bbox_coords = np.concatenate((bbox_coords, np.ones((8, 1))), 1)
    return bbox_coords

def project(intrinsic_mat,  # [3x3]
            pose_mat,  # [4x4], world coord -> camera coord
            box_coords  # [8x4] (x,y,z,1)
            ):
    """ Project 3D homogeneous coords to 2D coords. 
        in_front=True if at least 4 coords are in front of the camera. """

    #-- From world space to camera space.  
    box_coords_cam = pose_mat @ box_coords.T  # [4x8]
    box_coords[:3, :] /= box_coords[3, :]  # [8x4]
    
    #-- Handle points behind camera: negate x,y if z<0
    indices = box_coords_cam[2, :] < 0
    box_coords_cam[:2, indices] *= -1
    # box_coords_cam[:2, :] *= -1
    
    #-- From camera space to picture space. 
    box_coords_pic = intrinsic_mat @ box_coords_cam[:3, :]  # [3x8]
    # box_coords_pic = np.matmul(intrinsic_mat, box_coords_cam[:3, :])  # [3x8]
    box_coords_pic[:2, :] /= box_coords_pic[2, :]

    final_coords = np.array(np.transpose(box_coords_pic[:2, :], [1, 0]).astype(np.int16))

    # special for 3D front: y = height - y 
    final_coords[:, 1] = intrinsic_mat[1,2]*2 - final_coords[:, 1]

    #-- Report whether the whole object is in front of camera.
    in_front = True
    if np.sum(box_coords_cam[2,:]<0) < 8: # not in_front: less than 6 coords are in front of the camera
        in_front = False 

    fl_x, fl_y, cx, cy = intrinsic_mat[0,0], intrinsic_mat[1,1], intrinsic_mat[0,2], intrinsic_mat[1,2]
    angle_x = 2 * np.arctan(cx / fl_x)
    angle_y = 2 * np.arctan(cy / fl_y)
    theta_x = np.arctan2(box_coords_cam[0, :], -box_coords_cam[2, :])
    theta_y = np.arctan2(box_coords_cam[1, :], -box_coords_cam[2, :])
    
    if np.min(theta_x) > (angle_x/2) or np.max(theta_x) < (-angle_x/2) or \
        np.min(theta_y) > (angle_y/2) or np.max(theta_y) < (-angle_y/2):
        in_front = False

    return final_coords, in_front

def draw_bbox(img, pic_coords, color, label, width=4):
    """Draw bounding boxes and labels on the image. """
    if isinstance(img, np.ndarray):
        cv2.line(img, pic_coords[0], pic_coords[1], color, width)
        cv2.line(img, pic_coords[1], pic_coords[2], color, width)
        cv2.line(img, pic_coords[2], pic_coords[3], color, width)
        cv2.line(img, pic_coords[3], pic_coords[0], color, width)
        cv2.line(img, pic_coords[0], pic_coords[4], color, width)
        cv2.line(img, pic_coords[1], pic_coords[5], color, width)
        cv2.line(img, pic_coords[2], pic_coords[6], color, width)
        cv2.line(img, pic_coords[3], pic_coords[7], color, width)
        cv2.line(img, pic_coords[4], pic_coords[5], color, width)
        cv2.line(img, pic_coords[5], pic_coords[6], color, width)
        cv2.line(img, pic_coords[6], pic_coords[7], color, width)
        cv2.line(img, pic_coords[7], pic_coords[4], color, width)
        cv2.putText(img, label, (pic_coords[0][0], pic_coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    elif isinstance(img, Image):
        draw = ImageDraw.Draw(img)
        draw.line((*pic_coords[0], *pic_coords[1]), fill=color)
        draw.line((*pic_coords[1], *pic_coords[2]), fill=color)
        draw.line((*pic_coords[2], *pic_coords[3]), fill=color)
        draw.line((*pic_coords[3], *pic_coords[0]), fill=color)
        draw.line((*pic_coords[0], *pic_coords[4]), fill=color)
        draw.line((*pic_coords[1], *pic_coords[5]), fill=color)
        draw.line((*pic_coords[2], *pic_coords[6]), fill=color)
        draw.line((*pic_coords[3], *pic_coords[7]), fill=color)
        draw.line((*pic_coords[4], *pic_coords[5]), fill=color)
        draw.line((*pic_coords[5], *pic_coords[6]), fill=color)
        draw.line((*pic_coords[6], *pic_coords[7]), fill=color)
        draw.line((*pic_coords[7], *pic_coords[4]), fill=color)
        font = ImageFont.truetype(r'/usr/share/fonts/truetype/freefont/FreeMono.ttf', 20)
        draw.text(pic_coords[0], label, color, font)


def project_v2(intrinsic_mat,  # [3x3]
            pose_mat,  # [4x4], world coord -> camera coord
            box_coords  # [8x4] (x,y,z,1)
            ):
    #-- From world space to camera space.  
    box_coords_cam = pose_mat @ box_coords.T  # [4x8]
    box_coords[:3, :] /= box_coords[3, :]  # [8x4]
    
    box_coords_cam[:2, :] *= -1

    box_coords_cam = box_coords_cam.T

    
    return box_coords_cam[:, :3]

def draw_line(img, v1, v2, color, width, intrinsic_mat, clipping_plane_z):

    # box_coords_cam = np.stack([v1, v2], axis = 0)
    if -v1[2] > clipping_plane_z  and -v2[2] > clipping_plane_z: 
        # both points in front of clipping plane
        box_coords_cam = np.stack([v1, v2], axis = 0)

    elif -v1[2] < clipping_plane_z  and -v2[2] < clipping_plane_z: 
        # both points behind clipping plane
        return
    else:
        back_v = v2 if -v2[2] < clipping_plane_z else v1
        front_v  = v2 if -v2[2] > clipping_plane_z else v1
        
        # line function : r*t + b. We solve it when r_z*t + b_z = clipping_plane_z
        # Here r = front_v - back_v, b = back_v
        r = front_v - back_v
        b = back_v
        t = (-clipping_plane_z - b[2]) / r[2]
        x,y = r[0] * t +  b[0], r[1] * t +  b[1]
        intersect_v = np.array([x,y,-clipping_plane_z])
        box_coords_cam = np.stack([front_v, intersect_v], axis = 0)
    
    # print(box_coords_cam)
    # print("intrinsic={}".format(intrinsic_mat))
    
    
    box_coords_pic = intrinsic_mat @ box_coords_cam.T  # [3x2]
    # print(box_coords_pic)
    # print(box_coords_pic.shape)
    box_coords_pic[:2, :] /= box_coords_pic[2, :]
    # print(box_coords_pic)
    final_coords = box_coords_pic[:2, :].T.astype(np.int32)
    final_coords[:, 1] = intrinsic_mat[1,2]*2 - final_coords[:, 1]
    # final_coords[:, 0] = intrinsic_mat[0,2]*2 - final_coords[:, 0]

    # print("intrinsic={}".format(intrinsic_mat))
    # print("v1={}, v2={},\nrectified={}, final_coords={}".format(v1, v2, box_coords_cam, final_coords))
    
    cv2.line(img, final_coords[0], final_coords[1], color, width)
    

def draw_bbox_v2(img, cam_coords, color, label, intrinsic_mat, width=4, clipping_plane_z = 0.1):
    """Draw bounding boxes and labels on the image. """
    if isinstance(img, np.ndarray):
        draw_line(img, cam_coords[0], cam_coords[1], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[1], cam_coords[2], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[2], cam_coords[3], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[3], cam_coords[0], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[0], cam_coords[4], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[1], cam_coords[5], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[2], cam_coords[6], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[3], cam_coords[7], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[4], cam_coords[5], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[5], cam_coords[6], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[6], cam_coords[7], color, width, intrinsic_mat, clipping_plane_z)
        draw_line(img, cam_coords[7], cam_coords[4], color, width, intrinsic_mat, clipping_plane_z)
    else:
        print("GG.")


def project_aabb_to_image(img, # PIL image or numpy array
                          intrinsic_mat,  # [3x3]
                          pose,  # [4x4], world coord -> camera coord
                          aabb_codes,  # [Nx6]
                          labels="", # [n x str]
                          colors=None # a list of n tuples
                          ):
    """Project a list of bounding boxes to an image. Return the image with bounding boxes drawn. """
    img_with_bbox = img.copy()
    for aabb_code, label, color in zip(aabb_codes, labels, colors):
        bbox_coords = get_aabb_coords(aabb_code)
        pic_coords, in_front = project(intrinsic_mat, pose, bbox_coords)
        if in_front:
            draw_bbox(img_with_bbox, pic_coords, color, label)
    return img_with_bbox

def project_obb_to_image(img, # PIL image or numpy array
                          intrinsic_mat,  # [3x3]
                          pose,  # [4x4], world coord -> camera coord
                          obboxes,  # [Nx8x3]
                          labels=None, # [n x str]
                          colors=None, # a list of n tuples
                          line_width=4
                          ):
    """Project a list of bounding boxes to an image. Return the image with bounding boxes drawn. """
    img_with_bbox = img.copy()
    for i in range(len(obboxes)):
        obbox = obboxes[i]
        obbox = np.concatenate([obbox, np.ones([obbox.shape[0], 1])], axis=1) if obbox.shape[1] == 3 else obbox
        label = labels[i] if labels!=None and i<len(labels) else ""
        color = colors[i] if colors!=None and i<len(colors) else (0,0,255)
        pic_coords, in_front = project(intrinsic_mat, pose, obbox)

        if in_front:
            draw_bbox(img_with_bbox, pic_coords, color, label, line_width)
    return img_with_bbox



def project_obb_to_image_v2(img, # PIL image or numpy array
                          intrinsic_mat,  # [3x3]
                          pose,  # [4x4], world coord -> camera coord
                          obboxes,  # [Nx8x3]
                          labels=None, # [n x str]
                          colors=None, # a list of n tuples
                          line_width=4
                          ):
    """Project a list of bounding boxes to an image. Return the image with bounding boxes drawn. """
    img_with_bbox = img.copy()
    for i in range(len(obboxes)):
        obbox = obboxes[i]
        obbox = np.concatenate([obbox, np.ones([obbox.shape[0], 1])], axis=1) if obbox.shape[1] == 3 else obbox
        label = labels[i] if labels!=None and i<len(labels) else ""
        color = colors[i] if colors!=None and i<len(colors) else (0,0,255)
        cam_coords = project_v2(intrinsic_mat, pose, obbox)

        
        fl_x, fl_y, cx, cy = intrinsic_mat[0,0], intrinsic_mat[1,1], intrinsic_mat[0,2], intrinsic_mat[1,2]
        angle_x = 2 * np.arctan(cx / fl_x)
        angle_y = 2 * np.arctan(cy / fl_y)
        theta_x = np.arctan2(cam_coords[:, 0], -cam_coords[:, 2])
        theta_y = np.arctan2(cam_coords[:, 1], -cam_coords[:, 2])
        
        if np.min(theta_x) > (angle_x/2) or np.max(theta_x) < (-angle_x/2) or \
            np.min(theta_y) > (angle_y/2) or np.max(theta_y) < (-angle_y/2):
            continue

        draw_bbox_v2(img_with_bbox, cam_coords, color, label, intrinsic_mat, width = line_width,)
    return img_with_bbox
