COARSE_CONFIG="okutama_scenario2_coarse"
# CONFIG="rubble_c9_r4"

# train coarse global gaussian model
gpu_id=0
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python train_large.py --config config/$COARSE_CONFIG.yaml



# # train CityGaussian
# # obtain data partitioning
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python data_partition.py --config config/$CONFIG.yaml

