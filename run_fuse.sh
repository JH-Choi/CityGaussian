CONFIG="cat_ue_aerial_20_blockFcam"

# python convert.py -s data/UE-collected/aerial/block1_6
# python convert_cam.py -s data/UE-collected/aerial/blockF_20_cam

# python train_time.py --config config/$CONFIG.yaml
python render_fuse.py --config config/$CONFIG.yaml
python metrics.py -m output/$CONFIG