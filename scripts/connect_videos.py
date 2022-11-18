import os
os.environ['IMAGEIO_FFMPEG_EXE'] = '/Users/abraham/Desktop/fyp/RPN_NeRF_temp/ffmpeg'
from moviepy.editor import *
# from render_video import merge_videos
from os.path import join
import imageio
import cv2
from tqdm import tqdm
import numpy as np

def merge_videos(video1_path, video2_path, img_path, output_dir, text1, text2, text3):

    vid1 = imageio.get_reader(video1_path,  'ffmpeg')
    vid2 = imageio.get_reader(video2_path,  'ffmpeg')
    vid1_num_frames = vid1.count_frames()
    vid2_num_frames = vid2.count_frames()
    assert vid1_num_frames == vid2_num_frames, "The number of frames in two videos should be the same."

    bev_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w, _ = vid1.get_data(0).shape
    re_h = int(h * 2)
    re_w = int(re_h * 4 / 3)
    bev_w = re_w - w
    bev_h = int(bev_w * bev_img.shape[0] / bev_img.shape[1])
    bev_img_resize = cv2.resize(bev_img, (bev_w, bev_h), interpolation = cv2.INTER_AREA)

    vid_imgs = []
    for num in tqdm(range(vid2_num_frames)):
        img1 = vid1.get_data(num)
        img2 = vid2.get_data(num)

        h, w, _ = img1.shape
        re_h = int(h * 2)

        img = np.zeros((re_h, re_w, 3), dtype=np.uint8)
        img[0:h, -w:, :] = img1
        img[h:, -w:, :] = img2

        img[int((re_h-bev_h)/2):int((re_h-bev_h)/2)+bev_h, 0:bev_w, :] = bev_img_resize
        img = cv2.putText(img, text1, org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=2, color=(255,255,255), thickness=4, lineType=cv2.LINE_AA)
        img = cv2.putText(img, text2, org=(50, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=2, color=(255,255,255), thickness=4, lineType=cv2.LINE_AA)
        img = cv2.putText(img, text3, org=(50, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=2, color=(255,255,255), thickness=4, lineType=cv2.LINE_AA)

        vid_imgs.append(img)
        # break
    
    kargs = {'macro_block_size': None}
    writer = imageio.get_writer(join(output_dir, f'final_video.mp4'), fps=30, **kargs)
    for im in tqdm(vid_imgs):
        writer.append_data(im)
    writer.close()

video_output_dir = './video_output'
scene_list = [
              'Salon2_final',
              'asianRoom1_final',
              'asianRoom2_final',
              '3dfront_0089_00_final',
              '3dfront_0091_00_final', 
              '3dfront_0019_00_final', 
              'ai_001_008_final', 
              'ai_022_005_final', 
              'ai_053_020_final', 
              ]
print(scene_list)


clips = []
for scene_name in scene_list:
    print(scene_name)
    if '3dfront' in scene_name:
        dataset_type = '3dfront'
    elif 'ai_' in scene_name:
        dataset_type = 'hypersim'
    else:
        dataset_type = 'inria'
    
    if dataset_type == 'inria':
        text1 = "Inria NeRF Dataset"
        text3 = 'Type: Real-world'
    elif dataset_type == '3dfront':
        text1 = "3D-FRONT NeRF Dataset"
        text3 = 'Type: Synthetic'
    elif dataset_type == 'hypersim':
        text1 = "Hypersim NeRF Dataset"
        text3 = 'Type: Synthetic'
    text2 = "Scene: {}".format(scene_name[:-6])

    merge_videos(join(video_output_dir, scene_name, 'video_output.mp4'), 
                join(video_output_dir, scene_name, 'video_blend.mp4'), 
                join('./supmat_resources', scene_name[:-6], 'bev.png'),
                join(video_output_dir, scene_name), text1, text2, text3)
    
    clip = VideoFileClip(os.path.join(video_output_dir, scene_name,  'final_video.mp4'))
    clips.append(clip)

final = concatenate_videoclips(clips)
final.write_videofile(os.path.join(video_output_dir, 'final_video_connected.mp4'))

