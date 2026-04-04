### Git datasets for Q-RAG
```
workspace/
├── Q-RAG-1/
|  └── QRAG_hotpotqa_4090_eval_50
├── planner/     [one planner is enough]
|  ├── qwen_planner_lora_v2_musique_cleaned_v2   
|  └── qwen_planner_lora_v2                      
└── datasets/ 
   ├── hotptqa 
   └── musique
```
Python Environment
```bash
conda create -n qrag python=3.12 -y
conda activate qrag
python -m pip install -U pip wheel
pip install vllm  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets
pip install peft
# Check environment
python -c "from rl.agents.pqn import PQNActor; print('✅ Q-RAG installed successfully')"
```
### Download datasets (paste under the workspace folder)
```bash
cd workspace
git clone https://huggingface.co/datasets/Q-RAG/Hotpotqa_and_Musique
cd Hotpotqa_and_Musique
unzip hotpotqa+musique.zip -d /workspace/datasets
cd ..
rm -rf Hotpotqa_and_Musique
```
### Download planner (paste under the workspace folder)
```bash
mkdir planner
cd planner
# 第一版 musique数据集的planner
git clone https://huggingface.co/TroyHow/qwen_planner_lora_v2
# 第二版 修改musique数据集的planner
git clone https://huggingface.co/TroyHow/qwen_planner_lora_v2_musique_cleaned_v2
```
### Download Q-RAG repo and our trained model (paste under the workspace folder)
```bash
git clone -b xmum https://github.com/Patiskey/Q-RAG-1.git
cd Q-RAG-1
```
### Download our trained model (paste under the workspace/Q-RAG-1 folder)
```bash
# 加载我们训练的模型 （只需要他的eval_seed42.jsonl）
git clone https://huggingface.co/TroyHow/QRAG_hotpotqa_4090_eval_50
```

### Crtic Model RL
```bash
CUDA_VISIBLE_DEVICES=0,1 python Critic_rl_train.py \
--train_file ./curriculum_train.jsonl \
--planner_lora /workspace/planner/qwen_planner_lora_v2
```

### LLM Evaluation with Qwen2.5-7B-Instruct
```bash
python eval_llm_openqa_with_planner_chain_of_thought.py    \
--file_path ./QRAG_hotpotqa_4090_eval_50/eval_seed42.jsonl   \
--model_name Qwen/QwQ-32B   \
--planner_base Qwen/Qwen2.5-7B-Instruct    \
--planner_lora /workspace/planner/final    \
--output_file_path ./QRAG_hotpotqa_4090_eval_50/llm-answering_qwenplanner_eval.json
```
