model_name="regnetx_002"
python main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 0.01 \
     --output_dir output/${model_name}