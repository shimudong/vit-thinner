python vit_infer.py \
    --model-name "vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k" \
    --checkpoint-path timm/vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k/model.safetensors \
    --data-root imagenet_1k/val \
    --rank 780 \
    --batch-size 64 --num-workers 4
