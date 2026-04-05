# Plan_Q-RAG
## Setup Rent GPU
```
parent_dir/
├── Q-RAG/    
├── planner/     
|  ├── qwen_planner_lora_v2_musique_cleaned_v2   
|  └── qwen_planner_lora_v2                      
└── datasets/ 
|  ├── hotptqa 
|  └── musique
```
### Git datasets for Q-RAG
```bash
cd workspace
git clone https://huggingface.co/datasets/Q-RAG/Hotpotqa_and_Musique
cd Hotpotqa_and_Musique
unzip hotpotqa+musique.zip -d /workspace/datasets
cd ..
rm -rf Hotpotqa_and_Musique
du -h
```
### Git repo of Q-RAG
```bash
git clone https://github.com/griver/Q-RAG.git
cd Q-RAG
# 加载我们训练的模型 （只需要他的eval_seed）
git clone https://huggingface.co/TroyHow/QRAG_hotpotqa_4090_eval_50

#Only need when you don't have your self-trained hotpotqa model yet
git clone https://huggingface.co/Q-RAG/qrag-ft-e5-on-hotpotqa
```
### Environment Setup
```bash
# Setup venv
conda create -n qrag python=3.12 -y
conda activate qrag
python -m pip install -U pip wheel
pip install vllm  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets
pip install peft

# Check environment
python -c "from rl.agents.pqn import PQNActor; print('✅ Q-RAG installed successfully')"

```
### Train: Log with Time
original 100
```bash
python train_q_rag_logt.py \
   envs=hotpotqa \
   algo=pqn_e5_hotpotqa \
   envs.data_path="/workspace/datasets/hotpotqa" \
   steps_count=10000 \
   batch_size=12 \
   accumulate_grads=8 \
   eval_interval=50 \
   envs_parallel=1 \
   max_action_length=220
```
Force to use GPU 1
```bash
CUDA_VISIBLE_DEVICES=1 python train_q_rag_logt.py \
   envs=hotpotqa \
   algo=pqn_e5_hotpotqa \
   envs.data_path="/home/ai-faculty/workspace/datasets/hotpotqa" \
   steps_count=10000 \
   batch_size=12 \
   accumulate_grads=8 \
   eval_interval=50 \
   envs_parallel=1 \
   max_action_length=220
```
Zip for easier download
```bash
# Server
tar -cvf - outputs_folder | pigz -6 -p 32 | split -d -b 4G - models.tar.gz.
# Client
scp <username>@<ip_address>:/your/file/location/models.tar.gz.* D:\<your\file\location>
```
## E5 HotpotQA Retrievar Evaluation
E5 HotpotQA Retrievar Evaluation
```bash
python eval_retriever.py   \
   pretrained_path=./runs/QRAG_hotpotqa_4090_24h15m    \
   num_samples=-1    \
   +envs.max_steps=2    \
   +envs.data_path=/home/ai-faculty/workspace/datasets/hotpotqa
```
E5 HotpotQA Retiever Evaluation with musique
```bash
CUDA_VISIBLE_DEVICES=0 python eval_retriever.py \
  pretrained_path=./runs/QRAG_hotpotqa_4090_24h15m_50 \
  num_samples=-1 \
  +envs=musique \
  ++envs.max_steps=4 \
  ++envs.data_path=/home/ai-faculty/workspace/datasets/musique \
  +max_action_length=110 \
  +max_action_length_in_memory=110
```

## LLM Evaluation
LLM Evaluation HotpotQA Model Original technique
```bash
python eval_llm_openqa.py \
   --file_path ./runs/QRAG_hotpotqa_4090_24h15m/eval_seed42.jsonl \
   --model_name Qwen/QwQ-32B \
   --output_file_path ./runs/QRAG_hotpotqa_4090_24h15m/llm-answering_eval.json
```
LLM Evaluation with Meta-Llama-3.1-8B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0 python eval_llm_openqa_with_planner_chain_of_thought.py    \
--file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/eval_seed42.jsonl   \
--model_name Qwen/Qwen2.5-7B-Instruct    \
--planner_base meta-llama/Meta-Llama-3.1-8B-Instruct    \
--planner_lora /home/ai-faculty/workspace/planner/llama31_planner_lora_v1/final    \
--output_file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_llama31planner_eval.json
```
LLM Evaluation with Qwen2.5-7B-Instruct and CoT Retireval
```bash
CUDA_VISIBLE_DEVICES=0 python eval_llm_openqa_with_planner_chain_of_thought_v2.py    \
--file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/eval_seed42-old.jsonl   \
--model_name Qwen/Qwen2.5-7B-Instruct   \
--planner_base Qwen/Qwen2.5-7B-Instruct    \
--planner_lora /home/ai-faculty/workspace/planner/qwen_planner_lora_v2_musique_cleaned_v2/final    \
--output_file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_qwen-planner-clean_eval_CoT_Retires.json
```

### Original Train
```bash
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


## Computer resources / Test Results
[computer-resources_test-results.md](./computer-resources_test-results.md) 

## Server Environment Setup Guide
[Server.md](./Server.md)

## View Log in Table Format 
[log_table.md](./log_table.md)