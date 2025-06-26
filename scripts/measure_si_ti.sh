#!/bin/bash

DATA_SOURCE=../output_for_alignment_photogram_aligned
MAX_FRAMES=100 #${1:-100}

# scene → dataset list
declare -A SCENE_DATASETS=(
  # ["arena"]="arena_scene0" #arena_scene1 arena_scene2 arena_scene3 arena_scene4 arena_scene5 arena_scene6"
  #["couch"]="couch_scene0 couch_scene1  couch_scene2  couch_scene3"
  #["kitchen"]="kitchen_scene0 kitchen_scene1 kitchen_scene2 kitchen_scene3"
  #["whiteboard"]="whiteboard_scene0 whiteboard_scene1 whiteboard_scene2 whiteboard_scene3"
  ["mill19"]="mill19_scene0 mill19_scene1 mill19_scene2"
)

for SCENE in "${!SCENE_DATASETS[@]}"; do
  for DATA_NAME in ${SCENE_DATASETS[$SCENE]}; do
    echo "======================================="
    echo "Processing: ${SCENE}/${DATA_NAME}"

    echo "→ 2D SI/TI (${SCENE}/${DATA_NAME})"
    #python measure_2d_si_ti.py --scene_dir "${SCENE}/${DATA_NAME}" --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs
    echo
    echo "→ 3D SI/TI (${DATA_NAME})"
    python measure_3d_si_ti.py --scene_dir ${DATA_SOURCE}/${DATA_NAME}/gt/color_ptcl  --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs --unit m
    # python measure_mesh_si_ti.py --scene_dir ${DATA_SOURCE}/${DATA_NAME}/gt/mesh  --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}"
    #python measure_mesh_complexity_sequence.py --mesh_dir ${DATA_SOURCE}/${DATA_NAME}/gt/mesh --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs   
    echo    
  done
done

exit 1

MAX_CONCURRENT_JOBS=3  # Set the maximum number of concurrent jobs

for SCENE in "${!SCENE_DATASETS[@]}"; do
  for DATA_NAME in ${SCENE_DATASETS[$SCENE]}; do
    # Wait for a free slot if the number of running jobs reaches the limit
    while [[ $(jobs -r | wc -l) -ge $MAX_CONCURRENT_JOBS ]]; do
      sleep 1  # Wait for a free slot
    done

    # Process the dataset in the background
    {
      echo "======================================="
      echo "Processing: ${SCENE}/${DATA_NAME}"

      #echo "→ 2D SI/TI (${SCENE}/${DATA_NAME})"
      #python measure_2d_si_ti.py --scene_dir "${SCENE}/${DATA_NAME}" --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs
      echo
      echo "→ 3D SI/TI (${DATA_NAME})"
      python measure_3d_si_ti.py --scene_dir ${DATA_SOURCE}/${DATA_NAME}/gt/color_ptcl  --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs --unit m
      # python measure_mesh_si_ti.py --scene_dir ${DATA_SOURCE}/${DATA_NAME}/gt/mesh  --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}"
      # python measure_mesh_complexity_sequence.py --mesh_dir ${DATA_SOURCE}/${DATA_NAME}/gt/mesh --scene_name ${DATA_NAME} --max_frames "${MAX_FRAMES}" --output_dir ../ours_logs   
      echo
    } &
  done
done

echo "All measurements done."