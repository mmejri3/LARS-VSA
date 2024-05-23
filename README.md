# Abstractors

Welcome to the repository for the paper:

> "LARS-VSA: A Vector Symbolic Architecture For Learning with Abstract Rules"

This repository contains the following modules and directories:

- **Abstractor Modules**:
  - `abstracters.py` and `abstractor.py`: Implement different variants of the Abstractor module.
  - `autoregressive_abstractor.py`: Implements sequence-to-sequence abstractor-based architectures.
  - `seq2seq_abstracter_models.py`: An older, less general implementation of sequence-to-sequence models.

- **Attention Mechanisms**:
  - `multi_head_attention.py`: A modified version of TensorFlow's implementation, supporting various activation functions applied to attention scores.
  - `transformer_modules.py`: Implements different Transformer modules, including Encoders and Decoders.
  - `attention.py`: Implements various attention mechanisms for Transformers and Abstractors, including relational cross-attention.

- **Experiments**:
  - The `experiments` directory contains the code for all experiments presented in the paper. Each subdirectory contains a README with detailed instructions on how to replicate the experiments.

The Abstractor and the experiments are inspired by the work from [https://github.com/Awni00/abstractor.git](https://github.com/Awni00/abstractor.git).

For more detailed information on each module and the experiments, please refer to the individual READMEs within the respective directories.

