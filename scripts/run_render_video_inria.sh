python ./scripts/render_video.py \
--dataset_type inria \
--feature_dir ./inria_rpn_data/features \
--target_dir ./inria_rpn_results \
--output_dir ./video_output \
--single_scene asianRoom2 \
--hmp_type voxel \
--transpose_yz \
--hmp_top_k 70 \
--vis_top_n 10 \
--kernel_type box \
--value_scale 60 \
--view_angle 50 \
--downsample 1 \
--gaussian_sigma 4 \
--concat_img \
--blend_alpha_beta_gamma 0.6 0.5 0 \
--line_width 4 
