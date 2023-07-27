model_name="pit_b_224"
python main.py \
    --data-path "/home/aicourse_dataset/" \
    --model ${model_name} \
    --batch-size 32 \
    --epochs 50 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --test_only \
    --resume "output/${model_name}/best_checkpoint.pth" \
#     --lr 1e-4 \