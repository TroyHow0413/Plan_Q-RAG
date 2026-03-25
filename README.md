# Plan_Q-RAG
## Setup Rent GPU
```
parent_dir/
├── Q-RAG/      ← [Q-RAG](https://github.com/griver/Q-RAG.git)
└── datasets/   ← [datasets Hotpotqa and Musique](https://huggingface.co/datasets/Q-RAG/Hotpotqa_and_Musique)
```
### Git datasets for Q-RAG
```bash
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
## Sample DATA for checking
`train_q_rag.py` with Q-RAG original setting, 45:45 mins (NVIDIA A100-SXM4-80GB)

| Steps | Reward | Eval Reward | QF Loss |
| :----: | :----: | :----: | :----: |
| 800 | 0.43 | 0.343 | 2.95 |
| 1600 | 0.173 | 0.277 | 0.118 |
| 2400 | 0.401 | 0.713 | 0.187 |
| 3200 | 0.637 | 0.793 | 0.136 |

`train_q_rag.py` with `eval_interval = 50` setting, 48:22 mins (NVIDIA A100-SXM4-80GB)
| Steps | Reward | Eval Reward | QF Loss | Time Used (mins) |
| :----: | :----: | :----: | :----: | :----: |
| 800 | 0.402 | 0.367 | 2.14 | NA |
| 1200 | 0.167 | 0.197 | 1.12 | 19:15 |
| 1600 | 0.161 | 0.297 | 0.153 | 24:58 |
| 2000 | 0.267 | 0.507 | 0.106 | 30.21 |
| 2400 | 0.493 | 0.673 | 0.142 | 36:52 |
| 2800 | 0.587 | 0.713 | 0.157 | 43:00 |
| 3200 | 0.65 | 0.77 | 0.141 | 48:22 |

| Eval Interval | Steps | Reward | Eval Reward | QF Loss | Time Used (mins) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 50 | 800 | 0.402 | 0.367 | 2.14 | NA |
| 50 | 1200 | 0.167 | 0.197 | 1.12 | 19:15 |
| 50 | 1600 | 0.161 | 0.297 | 0.153 | 24:58 |
| 50 | 2000 | 0.267 | 0.507 | 0.106 | 30.21 |
| 50 | 2400 | 0.493 | 0.673 | 0.142 | 36:52 |
| 50 | 2800 | 0.587 | 0.713 | 0.157 | 43:00 |
| 50 | 3200 | 0.65 | 0.77 | 0.141 | 48:22 |
| 100 | 800 | 0.43 | 0.343 | 2.95 | NA |
| 100 | 1600 | 0.173 | 0.277 | 0.118 | NA |
| 100 | 2400 | 0.401 | 0.713 | 0.187 | NA |
| 100 | 3200 | 0.637 | 0.793 | 0.136 | NA |



## Computer resources
[基于HotpotQA+Musique(combined, GTE embedder) 训练出来的模型](https://huggingface.co/TroyHow/Q-RAG_Test/blob/main/QRAG_combined.zip) Q-RAG文中没有提及他的测试 <br>
- 训练时长：18:07:48
- 显卡： Pro 6000 96GB
- 显存占用：59GB ± 0.5GB
![结束的截图](./img/hotpotqa_mosique_combine_training.png)

HotpotQA_推理  
- 训练时长：00:12:26
- 显卡：NVIDIA A100-SXM4-80GB
- 显存占用：30GB ± 1GB
![结束的截图](./img/hotpotqa_original_Retriever_Evaluation.png)

LLM Evaluation: Original HotpotQA Model
- 训练时长：≈1h 10m
- 显卡：NVIDIA A100-SXM4-80GB
- 显存占用：60GB ± 0.5GB
![结束的截图](./img/hotpotqa_original_QwQ-32B_Evaluation.png)

HotpotQA Training With [Log with Time](./log_50_3h.txt) As REFERENCE
  eval_interval original 100
- 训练时长：3h 10m
- 显卡： NVIDIA A100-SXM4-80GB
- 显存占用：30GB ± 0.5GB (TBC)<br>
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
![结束的截图](./img/log_train_original_3h.png)
