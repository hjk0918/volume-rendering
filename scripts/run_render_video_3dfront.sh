python ./scripts/render_video.py \
--dataset_type 3dfront \
--feature_dir ./3dfront_rpn_data/features_160 \
--target_dir ./output/3dfront_fcos_density_only \
--output_dir ./video_output \
--hmp_type voxel \
--transpose_yz \
--hmp_top_k 70 \
--vis_top_n 10 \
--kernel_type box \
--value_scale 42 \
--view_angle 50 \
--downsample 1 \
--gaussian_sigma 4 \
--concat_img \
--blend_alpha_beta_gamma 0.6 0.5 0 \
--line_width 4 
