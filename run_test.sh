model_name="pit_b_224"
python main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 64 \
    --epochs 50 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --test_only \
    --resume "output/${model_name}/best_checkpoint.pth" \
#     --lr 1e-4 \