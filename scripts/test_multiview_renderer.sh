# Default values
DATA_DIR=../dataset/output_quest_2min
OUTPUT_DIR=../output
FRAME_COUNT=1 # must match with config.json
CAM_IDX_LIST="0 1 2 3" #2 3"

EXTRINSIC_SUFFIX="_scaniverse_aligned"
# EXTRINSIC_SUFFIX=""
INTRINSIC_PATH=../final_IMC_3D/intrinsics/arena_config.json
EXTRINSIC_PATH=../final_IMC_3D/extrinsics/global_extrinsics_arena${EXTRINSIC_SUFFIX}.npy


python multiview_renderer.py --cam_idx_list $CAM_IDX_LIST --data_path $DATA_DIR --mode gt --frame_count $FRAME_COUNT --output_dir ${OUTPUT_DIR} --reconstruct_geometry textured_mesh --extrinsic_path $EXTRINSIC_PATH --intrinsic_path $INTRINSIC_PATH \
    --scene_name kitchen