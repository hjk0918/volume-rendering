import os
import numpy as np
import json
import subprocess

scannet_dataset_path = '/Users/abraham/Desktop/fyp/RPN_NeRF_temp/scannet_dataset'
good_images_path = '/Users/abraham/Desktop/fyp/RPN_NeRF_temp/output/scannet_fcos_vggEF/scannet_test_results/good_images.txt'

keep_dict = {}

with open(good_images_path, 'r') as f:
    good_images = f.readlines()
    good_images = [x.strip() for x in good_images]

for image in good_images:
    split = image.split('_')
    scene_name = split[0]+'_'+split[1]
    image_name = split[2]
    if scene_name not in keep_dict:
        keep_dict[scene_name] = [image_name]
    else:
        keep_dict[scene_name].append(image_name)


for scene_name, image_names in keep_dict.items():
    frames = []
    with open(os.path.join(scannet_dataset_path, scene_name, 'transforms_test.json'), 'r') as f:
        test_dict = json.load(f)
    raw_frames = test_dict['frames']
    for image_name in image_names:
        # subprocess.run(['mv', str(os.path.join(scannet_dataset_path, scene_name, 'transforms_test.json')), 
        #                       str(os.path.join(scannet_dataset_path, scene_name, 'transforms_test_archive.json'))])
        for frame in raw_frames:
            if frame['file_path'].split('/')[-1].split('.')[0] == image_name:
                frames.append(frame)

    test_dict['frames'] = frames
    with open(os.path.join(scannet_dataset_path, scene_name, 'transforms_test_new.json'), 'w') as f:
        json.dump(test_dict, f, indent=4)
