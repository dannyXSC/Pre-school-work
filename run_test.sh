model_name="deit_tiny_patch16_224"
python main.py \
    --data-path "/home/aicourse_dataset/" \
    --model ${model_name} \
    --batch-size 32 \
    --epochs 20 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --test_only \
    --resume "output/${model_name}/checkpoint.pth" \
#     --lr 1e-4 \