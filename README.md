# BERT Fine-Tuning for Sentiment Analysis (SST-2)

This project fine-tunes **BERT-base-uncased** on the [GLUE SST-2 dataset](https://gluebenchmark.com/tasks) for binary sentiment classification (positive/negative).  
Training is performed in **Google Colab** using Hugging Face’s `transformers`, `datasets`, and `evaluate` libraries.

---------------------------------------------------------------------------------------------------------

## 🚀 Quick Links
- 📓 **Colab Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/llDanieLll/BERT_FineTuned/blob/main/bert_sst2_finetune.ipynb)
- 🤗 **Model on Hugging Face Hub**: *Coming soon*

---------------------------------------------------------------------------------------------------------

## 📦 Project Structure
BERT_FineTuned/
|-- bert_sst2_finetune.ipynb #Colab notebook for training/evaluation
|-- requirements.txt #Dependencies
|-- README.md # Project documentation

## 🛠️ Features
- Fine-tunes `bert-base-uncased` on SST-2 sentiment data
- Supports **GPU acceleration** on Colab (Tesla T4 or better)
- Easy training with Hugging Face Trainer API
- Model + tokenizer export with `save_pretrained()` for deployment

---------------------------------------------------------------------------------------------------------

## ⚙️ Setup & Installation

Clone this repository and install dependencies:
git clone https://github.com/llDanieLll/BERT_FineTuned.git
cd BERT_FineTuned
pip install -r requirements.txt

🏋️‍♂️ Training
Inside the Colab notebook, simply run:
trainer.train()
trainer.evaluate()

💾 Saving and Loading the Model
# Save
save_dir = "bert-sst2-t4-best"

trainer.model.save_pretrained(save_dir)

tok.save_pretrained(save_dir)

# Load
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(save_dir)

tok = AutoTokenizer.from_pretrained(save_dir)

# 🔍 Inference Example
from transformers import TextClassificationPipeline 

pipe = TextClassificationPipeline(model=model, tokenizer=tok, device=0) 

print(pipe("This movie was absolutely fantastic!")) 

Output: [{'label': 'LABEL_1', 'score': 0.9999}]

# 📈 Next Steps
Upload trained weights to Hugging Face Hub

Try DistilBERT for faster inference

Fine-tune on custom datasets (CSV/JSON)

Experiment with LoRA/PEFT for lightweight fine-tuning

#📜 License

MIT License – feel free to use, modify, and share.

#👨‍💻 Author

Daniel Qiu

GitHub
