### Git datasets for Q-RAG
```bash
cd workspace
git clone https://huggingface.co/datasets/Q-RAG/Hotpotqa_and_Musique
cd Hotpotqa_and_Musique
unzip hotpotqa+musique.zip -d /workspace/datasets
cd ..
rm -rf Hotpotqa_and_Musique

git clone -b xmum https://github.com/Patiskey/Q-RAG-1.git
cd Q-RAG
# 加载我们训练的模型 （只需要他的eval_seed）
git clone https://huggingface.co/TroyHow/QRAG_hotpotqa_4090_eval_50


conda create -n qrag python=3.12 -y
conda activate qrag
python -m pip install -U pip wheel
pip install vllm  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core tensorboard rotary-embedding-torch pandas nltk sortedcontainers accelerate datasets
pip install peft

# Check environment
python -c "from rl.agents.pqn import PQNActor; print('✅ Q-RAG installed successfully')"

# 手动拉 Planer 权重 到 /workspace/planner/final 目录下
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
