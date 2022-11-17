import os
import cv2
import numpy as np

def main():
    input_path = '3dfront_vis/19_raw_2_input.png'
    hmp_path = '3dfront_vis/19_raw_2_hmp.png'
    output_dir = '3dfront_vis/blend'
    os.makedirs(output_dir, exist_ok=True)

    input = cv2.imread(input_path)
    hmp = cv2.imread(hmp_path)

    blend_img = cv2.addWeighted(input, 0.5, hmp, 0.5, 20)
    cv2.imwrite('3dfront_vis/19_raw_2_blend.png', blend_img)

    # for alpha in np.linspace(0.0, 1.0, 11):
    #     blend_img = cv2.addWeighted(input, alpha, hmp, 1-alpha, 10)
    #     cv2.imwrite(os.path.join(output_dir, 'blend_{:.1f}.png'.format(alpha)), blend_img)

if __name__ == '__main__':
    main()

