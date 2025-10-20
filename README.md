<h1 align="center">
Ai-Assisted Q&A And Diagnosis Of Thyroid Cytopathology
</h1>

##  Overview 
An Intelligent Thyroid Pathology Diagnosis Framework Integrating Large Language Models and Dynamic Knowledge Graphs. Through multi-source data fusion and context-aware visual reasoning, this framework delivers precise, interpretable, and efficient diagnostic solutions, aiming to enhance clinical diagnostic confidence and promote healthcare equity.

![Thyroid QA300](https://github.com/JtingF101/ThyroidDiagnosis_QA_KGArevion/blob/main/dataset/README_pic/ThyroidMQ-300.png)

##   Installation

First, clone the Github repository:

```bash
git clone https://github.com/JtingF101/ThyroidDiagnosis_QA_KGArevion/
```

Next, set up the environment. To create an environment containing all required packages, it's best to use conda (depending on your deployment conditions), then execute the following commands:

```bash
conda create -n thyroidQA python=3.9 -y
conda activate thyroidQA
pip install requirements.txt 
#The first deployment may encounter dependency version conflicts.
```

##  Running

After cloning the repository and installing all dependencies. 
- Ensure directory structure
```bash
ThyroidDiagnosis_QA_KGArevion/
│
├── dataset/
│   └── benchmark.json        #Questions
│
├── bioGPT.py             
│
└── Llama-original.py 
│
└── RAG-LLM.py 
```
- Navigate to the project directory in the terminal (ensure the current path is correct):
```bash
cd path/to/ThyroidDiagnosis_QA_KGArevion
python bioGPT.py
python RAG-LLM.py
python Llama-original.py
```

To run thyroidQA_KGARevion directly:
- First, please download the finetuned checkpoints (files denoted by *finetuned model*, including *adapter_config.bin*, *adapter_model.bin*, *embeddings.pth*, *lm_to_kg.pth*) which is available at [https://drive.google.com/drive/folders/1U5oT37bMocRYgobDFc-peFqS3WqE6zZs?usp=sharing], to 'thyroid-fintune' folder.
- Second, please run KGARevion-Thyroid.py by the following command:
   
```bash
python KGARevion-Thyroid.py --dataset thyroid
                            --max_round 2 
                            --is_revise True 
                            --llm_name llama3.1
                            --type MCQ
                            --weights_path thyroid-fintune
```
* dataset: Indicates that different QA datasets can be swapped out for testing.
* max_round: Indicates the maximum number of revision rounds
* is_revise: Indicates whether KGARevision includes revision operations
* llm_name: Indicates the backbone large language model of KGARevision
* type: Indicates the question-answering type
* weights_path: Indicates the weight path of KGARevision's backbone large language model

![KGARevion_Thyroid framework](https://github.com/JtingF101/ThyroidDiagnosis_QA_KGArevion/blob/main/dataset/README_pic/Framwork.png)
