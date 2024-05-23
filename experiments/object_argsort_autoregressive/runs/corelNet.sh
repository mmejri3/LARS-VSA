python3.8 evaluate_argsort_model_learning_curves.py --model CorelNet --pretraining_mode "none" --eval_task_data_path "object_sorting_datasets/product_structure_reshuffled_object_sort_dataset.npy" --n_epochs 150 --early_stopping True --min_train_size 10 --max_train_size 510 --train_size_step 50 --num_trials 2 --start_trial 0 --pretraining_train_size -1 --wandb_project_name "Sorting-5" 

