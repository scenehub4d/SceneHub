#!/bin/bash

DATA_SOURCE=/shiraz/MM_release
CAM_POSE_SOURCE=${DATA_SOURCE}/camera_pose
OUTPUT_SOURCE=../output

run_pipeline() {
    local SCENE=$1
    local DATA_NAME=$2
    local FRAME_COUNT=1

    if [[ "$DATA_NAME" == *_scene0 ]]; then
        FRAME_COUNT=1
        echo "Using 1 frame for ${DATA_NAME}"
    fi

    local DATA_DIR=${DATA_SOURCE}/rgbd_data/${SCENE}/${DATA_NAME}/rgbd
    local INTRINSIC_PATH=${CAM_POSE_SOURCE}/intrinsics/${SCENE}_config.json
    local EXTRINSIC_PATH=${CAM_POSE_SOURCE}/extrinsics/extrinsics_${SCENE}.npy


    if [[ "$DATA_NAME" == "arena_scene0" ]]; then
        EXTRINSIC_PATH=${CAM_POSE_SOURCE}/extrinsics/extrinsics_arena_static.npy
    fi

    local OUTPUT_DIR=${OUTPUT_SOURCE}/${DATA_NAME}
    mkdir -p ${OUTPUT_DIR}

    echo "Running reconstruct for ${DATA_NAME}"
    
    cd ../experiments

    python reconstruct.py --rgbd_path ${DATA_DIR} \
        --frame_count ${FRAME_COUNT} --output_dir ${OUTPUT_DIR} \
        --reconstruct_geometry color_ptcl \
        --intrinsic_path ${INTRINSIC_PATH} --extrinsic_path ${EXTRINSIC_PATH}

    python reconstruct.py --rgbd_path ${DATA_DIR} \
        --frame_count ${FRAME_COUNT} --output_dir ${OUTPUT_DIR} \
        --reconstruct_geometry textured_mesh \
        --intrinsic_path ${INTRINSIC_PATH} --extrinsic_path ${EXTRINSIC_PATH}

    #python o3d_player.py --data_path ${OUTPUT_DIR}/gt/color_ptcl
}

# Declare scenes and associated data names
declare -A SCENE_DATASETS
SCENE_DATASETS["arena"]="arena_scene0 arena_scene1 arena_scene2 arena_scene3 arena_scene4 arena_scene5 arena_scene6"
SCENE_DATASETS["couch"]="couch_scene0 couch_scene1 couch_scene2 couch_scene3"
SCENE_DATASETS["kitchen"]="kitchen_scene0 kitchen_scene1 kitchen_scene2 kitchen_scene3"
SCENE_DATASETS["whiteboard"]="whiteboard_scene0 whiteboard_scene1 whiteboard_scene2 whiteboard_scene3"
SCENE_DATASETS["mill19"]="mill19_scene0 mill19_scene1 mill19_scene2"

for SCENE in "${!SCENE_DATASETS[@]}"; do

    for DATA_NAME in ${SCENE_DATASETS[$SCENE]}; do
        while [[ $(jobs -r | wc -l) -ge 1 ]]; do
            sleep 1  # Wait for a free slot
        done
        run_pipeline "$SCENE" "$DATA_NAME" &
    done
done
wait  # Wait for all background jobs to finish
echo "All pipelines completed."