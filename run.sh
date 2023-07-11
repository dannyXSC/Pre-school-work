model_name="tnt_s_patch16_224"
python main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 32 \
    --epochs 50 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \




#    --flip \
#    --rotation


# tnt_s_patch16_224 swin_tiny_patch4_window7_224