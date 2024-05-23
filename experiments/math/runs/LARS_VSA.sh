list_subject=( 'comparison__kth_biggest')

for name in "${list_subject[@]}"; do
    j=0
    for i in $(seq 10 500 5000); do
        CUDA_VISIBLE_DEVICES=0 python train_model.py --model larsvsa --task "$name" --model_size small --run_name "trial=2" --n_epochs 150 --batch_size 16 --train_size $i --wandb_project_name "$name"
        ((j++))
    done
done
