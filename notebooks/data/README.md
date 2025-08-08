
---
pretty_name: "MoodPulse: Processed Data and Embeddings for Emotion Analysis"
license: mit
language:
- en
tags:
- emotion-classification
- affective-computing
- text-classification
- goemotions
- distilbert
- embeddings
task_categories:
- text-classification
dataset_info:
  source_dataset: "GoEmotions"
  includes:
    - raw data
    - tokenized data
    - transformer embeddings
  processed_by: "AffectiveLens pipeline"
---

# 📊 MoodPulse: Processed Data and Embeddings for Emotion Analysis

**MoodPulse** provides a self-contained dataset repository for use with the [AffectiveLens](https://github.com/your-username/AffectiveLens) project—an end-to-end NLP pipeline for emotion detection in text. It includes the full processing stack from raw text to final DistilBERT-based sentence embeddings, allowing researchers to bypass time-consuming preprocessing and directly train or benchmark models.

---

## 🧾 Dataset Description

This dataset builds upon the original **[GoEmotions](https://github.com/google-research/goemotions)** dataset by Google Research, which includes 58k carefully curated Reddit comments labeled with 28 fine-grained emotions.

In **MoodPulse**, these labels are condensed into **three mutually exclusive emotion classes**:
- Positive
- Neutral
- Negative

The dataset is structured to support every phase of the AffectiveLens pipeline:
- Raw CSVs
- Tokenized data in Hugging Face `datasets` format
- Precomputed `DistilBERT` embeddings

This enables full reproduction of results without requiring re-tokenization or embedding computation.

---

## 🗂️ Dataset Structure

The dataset is organized into logical folders corresponding to different stages of processing:

```

/
├── data/
│   ├── full\_dataset/
│   │   ├── goemotions\_1.csv
│   │   ├── goemotions\_2.csv
│   │   └── goemotions\_3.csv
│   │
│   ├── processed/
│   │   ├── GoEmotions\_Tokenized\_Train\_Pool/
│   │   └── GoEmotions\_Tokenized\_Test/
│   │
│   └── embeddings/
│       ├── MentalTrain/
│       └── MentalTest/

````

### 📁 Folder Descriptions

- **`data/full_dataset/`**  
  Original GoEmotions CSV files split into parts.

- **`data/processed/`**  
  Tokenized datasets using Hugging Face `datasets` format, ready for embedding extraction.

- **`data/embeddings/`**  
  Final DistilBERT `[CLS]` token embeddings for the training and test sets. These are saved as Hugging Face datasets and ready for model input.

---

## 🚀 How to Use

You can load the tokenized data or precomputed embeddings directly using the Hugging Face `datasets` library.

```python
from datasets import load_dataset

# Define repository ID and folder to load
repo_id = "psyrishi/MoodPulse"
data_folder = "data/embeddings/MentalTrain"  # or "data/embeddings/MentalTest"

# Load the dataset split
train_embeddings = load_dataset(repo_id, data_dir=data_folder, split='train')

print("Sample entry:")
print(train_embeddings[0])

# Access embeddings and labels
embedding_vector = train_embeddings[0]['cls_embedding']
label_vector = train_embeddings[0]['labels']
````

> 💡 Tip: You can replace `data_dir` to load the tokenized datasets instead, if desired.

---

## 📌 Use Cases

* Train or benchmark emotion classification models using high-quality, preprocessed embeddings.
* Compare performance of traditional ML models vs. transformer-based models.
* Build emotion-aware applications for mental health, customer feedback, or social media monitoring.

---

## 📚 Citation

This dataset is a **processed derivative** of the original GoEmotions dataset:

```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2020}
}
```

If you use **MoodPulse** in your work, please cite both the original GoEmotions authors and link back to this repository.

---

## ⚖️ Licensing

* **Original data**: Provided under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license by Google Research.
* **Code and processing logic**: Provided under the **MIT License**.

Please refer to the [LICENSE](./LICENSE) file for full details.

---

## 🙏 Acknowledgments

Special thanks to Google Research for the creation and open release of the GoEmotions dataset, and to the Hugging Face team for providing the open-source tools that made this processing pipeline possible.

---

## 🔗 Related Projects

* [GoEmotions Dataset (Google)](https://github.com/google-research/goemotions)
* [AffectiveLens](https://github.com/psywarrior1998/AffectiveLens) — Emotion detection pipeline built on top of this dataset.

---
