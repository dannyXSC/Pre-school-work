model_name="FixMatch"
python FixMatch_main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 8 \
    --epochs 50 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --rotation 45 \
    --flip 0.5 \
    --add_origin_image \
    --mu 4 \
    --threshold 0.95 \
    --eval-step 1024 \
#    --total-steps 2**20  \

