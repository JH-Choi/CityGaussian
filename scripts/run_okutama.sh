# get_available_gpu() {
#   local mem_threshold=500
#   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
#   $2 < threshold { print $1; exit }
#   '
# }


COARSE_CONFIG="okutama_s2_coarse"
CONFIG="okutama_s2_c4"

# out_name="val"  # i.e. TEST_PATH.split('/')[-1]
max_block_id=3  # i.e. x_dim * y_dim * z_dim - 1
port=4041

# train coarse global gaussian model
# gpu_id=0
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python train_large.py --config config/$COARSE_CONFIG.yaml


# train CityGaussian
# obtain data partitioning
# gpu_id=$(get_available_gpu)
gpu_id=0
# echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python data_partition.py --config config/$CONFIG.yaml

# # optimize each block, please adjust block number according to config
# for num in $(seq 0 $max_block_id); do
#     while true; do
#         # gpu_id=$(get_available_gpu)
#         gpu_id=0
#         if [[ -n $gpu_id ]]; then
#             echo "GPU $gpu_id is available. Starting training block '$num'"
#             # CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $num --port $port &
#             CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python train.py --config config/$CONFIG.yaml --block_id $num --port $port &
#             # Increment the port number for the next run
#             ((port++))
#             # Allow some time for the process to initialize and potentially use GPU memory
#             sleep 120
#             break
#         else
#             echo "No GPU available at the moment. Retrying in 2 minute."
#             sleep 120
#         fi
#     done
# done
# wait

