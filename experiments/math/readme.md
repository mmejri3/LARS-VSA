# Mathematical problem-solving

This set of experiments evaluates Abstractor architectures on a set of mathematical problem-solving tasks from the [`mathematics_dataset`](https://github.com/google-deepmind/mathematics_dataset) contributed by Saxton, Grefenstette, Hill, and Kohli.

Steps to reproduce experiments:

For each `task`, `model`, `model_size`, `trial`, etc., run the following:
```
python train_model.py --model {model} --task {task} --model_size {model_size} --run_name trial={trial} --n_epochs 50 --batch_size 128 --train_size -1
```

 We evaluate the following `model`'s: `['relational_abstractor', 'transformer', 'lars_vsa']`. For the Abstractor, we evaluate a `model_size` of `'small'`. For the Transformer, we evaluate `model_size`'s `['small']`.


