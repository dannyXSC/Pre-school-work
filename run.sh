model_name="pit_b_224"
python main.py \
    --data-path "/home/aicourse_dataset/" \
    --model ${model_name} \
    --batch-size 32 \
    --epochs 50 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --rotation 45 \
    --flip 0.5 \
    --add_origin_image \
    --split_dataset \
# pit_b_224
