python imagenet_compressed_sensing.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/compressed_sensing_config.yaml \
    --timestep=20 \
    --cs_ratio=0.5 \
    --method="mpgd_admmp" \

python imagenet_compressed_sensing.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/compressed_sensing_config.yaml \
    --timestep=20 \
    --cs_ratio=0.3 \
    --method="mpgd_admmp" \

python imagenet_compressed_sensing.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/compressed_sensing_config.yaml \
    --timestep=20 \
    --cs_ratio=0.1 \
    --method="mpgd_admmp" \
