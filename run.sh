model_name="cait_xxs24_224"
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
    --epoch_per_print 10 \
#    --split_dataset \
# pit_b_224
# set1 cifar100
# set2 country211