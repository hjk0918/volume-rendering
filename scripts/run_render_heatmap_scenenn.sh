python ./scripts/render_heatmap.py \
--dataset_dir ./scenenn_dataset \
--feature_dir ./scenenn_rpn_data/features_160 \
--target_dir ./output/3dfront_fcos_vggEF \
--output_dir ./output/3dfront_fcos_vggEF/scenenn \
--boxes_dir ./scenenn_rpn_data/boxes_160_aabb \
--single_scene 025 \
--hmp_type voxel \
--transpose_yz \
--hmp_top_k 70 \
--vis_top_n 6 \
--kernel_type box \
--value_scale 35 \
--view_angle 45 \
--downsample 1 \
--gaussian_sigma 4 \
--concat_img \
--blend_alpha_beta_gamma 0.6 0.5 0 \
--line_width 4 \
--command_path ./scripts/run_render_heatmap_scenenn.sh
# --use_gt


# python ./scripts/render_heatmap.py \
# --dataset_dir ./FRONT3D_render/test_scenes \
# --feature_dir ./3dfront_rpn_data/features_160 \
# --target_dir ./output/3dfront_vggEF_from_scratch_aug \
# --output_dir ./heatmap_output/3dfront_vggEF_from_scratch_aug \
# --boxes_dir ./3dfront_rpn_data/boxes_160_obb \
# --transpose_yz \
# --top_n 8 \
# --kernel_type box \
# --value_scale 30 \
# --downsample 2 \
# --gaussian_sigma 1 \
# --concat_img \
# --use_voxel