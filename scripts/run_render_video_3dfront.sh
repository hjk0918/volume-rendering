python ./scripts/render_video.py \
<<<<<<< HEAD
--dataset_type 3dfront \
=======
--dataset_dir ./3dfront_dataset \
>>>>>>> 8fd585d9998785a7c2f78e8ece97a9a3e66d0467
--feature_dir ./3dfront_rpn_data/features_160 \
--target_dir ./3dfront_rpn_results \
--output_dir ./video_output \
--single_scene 3dfront_0091_00 \
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
