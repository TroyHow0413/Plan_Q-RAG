# Computer resources / Test Results
## Retrievar Q-RAG Original
### HotpotQA Retrievar Evaluation  
- 时长：00:12:26
- 显卡：NVIDIA A100-SXM4-80GB
- 显存占用：30GB ± 1GB
![结束的截图](./img/hotpotqa_original_Retriever_Evaluation.png)

### LLM Evaluation: Original HotpotQA Model
- 时长：1h 10m
- 显卡：NVIDIA A100-SXM4-80GB
- 显存占用：60GB ± 0.5GB
![结束的截图](./img/hotpotqa_original_QwQ-32B_Evaluation.png)

### HotpotQA Training With [Log with Time](./log_50_3h.txt) As REFERENCE
`eval_interval=100`
- 训练时长：3h 10m
- 显卡： NVIDIA A100-SXM4-80GB
- 显存占用：31GB ± 0.5GB (TBC)<br>
![结束的截图](./img/log_train_original_3h.png)

## Q-RAG rerun on 4090D
### HotpotQA Training With [4090D Log with Time](./log_50_4090_full.txt) As REFERENCE
`eval_interval=50`
- 训练时长：24:14:16
- 显卡： NVIDIA 4090D 48GB
- 显存占用：31.7GB ± 0.5GB 
![结束的截图](./img/hotpotqa_4090_50_24h15m.png)

### HotpotQA Retrievar Evaluation 
- 时长：00:12:26
- 显卡：NVIDIA 4090D 48GB
- 显存占用：30GB ± 0.5GB
![结束的截图](./img/hotpotqa_4090_50_24h15m_Retrievar_Evaluation.png)

### LLM Evaluation: 4090D HotpotQA Model (QwQ-32B)
- 时长：1h 10m
- 显卡：NVIDIA A100-SXM4-80GB
- [显存占用](./img/LLM_Evaluation_VRAM.png)：79.6 ± 0.1GB
![结束的截图](./img/hotpotqa_4090_50_24h15m_LLM_Evaluation.png)


## Planner Evaluation: Train on 4090D
### Musique+Qwen2.5-7B-Instruct Planner QA LLM Evaluation
Planner：Qwen2.5-7B-Instruct
- train on musiqu cleaned dataset
QA：Qwen2.5-7B-Instruct
- gpu memory utilization 0.75 (if 0.95 will OOM)
Computer resources:
- 时长：5-6 hours 
- 显卡：NVIDIA 4090D 48GB
![结束的截图](./img/qwen_musique-clean_planner.png)

## HotpotQA Planner Evaluation: Train on 4090D
### HotpotQA Planner Evaluation
Planner：qwen2.5 7b instruct
- train on musiqu （original datasets）
QA：qwen2.5 7b instruct
- gpu memory utilization 0.75 (if 0.95 will OOM)
Computer resources:
- 时长：00:12:26
- 显卡：NVIDIA 4090D 48GB
![结束的截图](./img/qwen_musique-ori_planner.png)





### HotpotQA+Musique Train
[基于HotpotQA+Musique(combined, GTE embedder) 训练出来的模型](https://huggingface.co/TroyHow/Q-RAG_Test/blob/main/QRAG_combined.zip) Q-RAG文中没有提及他的测试 <br>
- 训练时长：18:07:48
- 显卡： Pro 6000 96GB
- 显存占用：59GB ± 0.5GB
![结束的截图](./img/hotpotqa_mosique_combine_training.png)