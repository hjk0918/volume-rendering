
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:ZSW
@file:picture_compress.py
@time:2020/08/06
"""
 
import base64
import io
import os
from PIL import Image
from PIL import ImageFile
import cv2
import glob
import os
 
# newpath = 'D:\picture_test\picture_test'
# if not os.path.exists(newpath):
#     os.mkdir(newpath)
 
# 压缩图片文件
def compress_image(outfile, mb=190, quality=85, k=0.9):
    """不改变图片尺寸压缩到指定大小
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标,KB   190kb
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = os.path.getsize(outfile) // 1024
    print(o_size, mb)
    if o_size <= mb:
        print(outfile)
        print(outfile.split('.')[0] + '.png')
        return outfile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    while o_size > mb:
        im = Image.open(outfile)
        x, y = im.size
        out = im.resize((int(x * k), int(y * k)), Image.ANTIALIAS)
        try:
 
            out.save(outfile, quality=quality)
        except Exception as e:
            print(e)
            break
        o_size = os.path.getsize(outfile) // 1024
    return outfile
 
 
# 压缩base64的图片
def compress_image_bs4(b64, mb=190, k=0.9):
    """不改变图片尺寸压缩到指定大小
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标,KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    f = base64.b64decode(b64)
    with io.BytesIO(f) as im:
        o_size = len(im.getvalue()) // 1024
        if o_size <= mb:
            return b64
        im_out = im
        while o_size > mb:
            img = Image.open(im_out)
            x, y = img.size
            out = img.resize((int(x * k), int(y * k)), Image.ANTIALIAS)
            im_out.close()
            im_out = io.BytesIO()
            out.save(im_out, 'jpeg')
            o_size = len(im_out.getvalue()) // 1024
        b64 = base64.b64encode(im_out.getvalue())
        im_out.close()
        return str(b64, encoding='utf8')
 
#将指定文件夹filePath下的 文件地址 和 子文夹下的文件地址 塞进picuture_list列表中
def read_file(filePath):
    picuture_list = []
    for dirpath, dirnames, filenames in os.walk(filePath):
        path = [os.path.join(dirpath, names) for names in filenames]
        picuture_list.extend(path)
    return picuture_list

def find_imgs(root, img_list):

    img_list += glob.glob(os.path.join(root, '*.png'))
    
    item_list = os.listdir(root)
    for item in item_list:
        if os.path.isdir(os.path.join(root, item)):
            find_imgs(os.path.join(root, item), img_list)
 
if __name__ == '__main__':
    img_list = []
    root_dir = './figs'
    find_imgs(root_dir, img_list)
    
    for png_path in img_list:
        compress_image(png_path, mb=100)   #先压缩图片
        # jpg_to_png(file)   #再将图片转为PNG格式
