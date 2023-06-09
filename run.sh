model_name="cait_xxs36_224"
# train
python main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 0.01 \
<<<<<<< HEAD
    --output_dir output/${model_name}
&&
# test
python main.py \
    --data-path "/remote-home/share/course23/aicourse_dataset_final/" \
    --model ${model_name} \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --output_dir output/${model_name} \
    --test_only \
    --resume "output/${model_name}/best_checkpoint.pth"


=======
    --output_dir output/${model_name}
>>>>>>> af2e27a18a641bdd0e4f3632568d72d2c77880f5
