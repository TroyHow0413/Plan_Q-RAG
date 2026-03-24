# Plan_Q-RAG

## Setup Rent GPU
Git Clone to all the required data
```bash

git clone https://github.com/griver/Q-RAG.git
cd Q-RAG
#Only need when you don't have your self-trained model
git clone https://huggingface.co/Q-RAG/qrag-ft-e5-on-hotpotqa
```
Environment Setup
```bash
# Setup venv
conda create -n qrag python=3.12 -y
conda activate qrag

python -m pip install -U pip wheel
pip install vllm  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets

# Check environment
python -c "from rl.agents.pqn import PQNActor; print('✅ Q-RAG installed successfully')"

```

```bash
cd ..

python train_q_rag_logt.py \
   envs=hotpotqa \
   algo=pqn_e5_hotpotqa \
   envs.data_path="/workspace/datasets/hotpotqa" \
   steps_count=10000 \
   batch_size=12 \
   accumulate_grads=8 \
   eval_interval=50 \ #original 100
   envs_parallel=1 \
   max_action_length=220

python train_q_rag.py \
   envs=hotpotqa \
   algo=pqn_e5_hotpotqa \
   envs.data_path="/workspace/datasets/hotpotqa" \
   steps_count=10000 \
   batch_size=12 \
   accumulate_grads=8 \
   eval_interval=100\
   envs_parallel=1 \
   max_action_length=220

```
