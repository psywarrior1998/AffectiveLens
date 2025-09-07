# AffectiveLens: A Transformer-Based Engine for Emotion Detection

AffectiveLens is an end-to-end AI project for emotion detection. This system uses state-of-the-art DistilBERT Transformer embeddings to understand text nuances. It trains and benchmarks 6 machine learning models to accurately classify sentiment as positive, negative, or neutral, demonstrating a complete and robust NLP pipeline.

## 📜 Project Overview

This project implements a full data science pipeline to tackle the problem of emotion detection in text, a key task in the field of Affective Computing and Natural Language Processing. It leverages the state-of-the-art DistilBERT model from Hugging Face to generate rich, contextual embeddings from raw text data. These embeddings serve as high-quality features to train and evaluate a suite of powerful classifiers—from Logistic Regression and SVMs to advanced gradient-boosted ensembles like LightGBM and XGBoost—to find the optimal model for the task.

The final result is a robust classifier that can accurately predict the emotional valence of new, unseen text, along with a deployable prediction function that encapsulates the entire workflow.

-----

## 🚀 Live Demo

Experience the AffectiveLens engine in action\!

You can test the live, deployed application by clicking the link below:

**[Try the AffectiveLens App on Streamlit\!](https://affectivelens.streamlit.app/)**

-----

## ✨ Key Features

  - **State-of-the-Art Embeddings**: Utilizes the `distilbert-base-uncased` model for high-quality contextual text representation.
  - **Comprehensive Model Comparison**: Trains and evaluates 6 different machine learning algorithms in a "gauntlet" style to empirically determine the best performer.
  - **Robust Data Processing**: Includes a full pipeline for cleaning, balancing (via oversampling), and preparing the dataset for modeling.
  - **End-to-End Workflow**: The project is a self-contained notebook demonstrating the entire lifecycle from data ingestion to a final, deployable prediction function.
  - **Reproducible Environment**: The notebook includes a setup cell to ensure a consistent and reproducible software environment.

## ⚙️ Project Architecture & Workflow

The project follows a systematic, multi-stage pipeline:

1.  **Environment Setup**: A sterile, version-pinned environment is created to ensure full reproducibility.
2.  **Data Ingestion**: Pre-computed embeddings for the GoEmotions dataset are automatically downloaded from the project's Hugging Face repository.
3.  **Data Preparation**: The data is split into training, validation, and test sets. The 28 original emotion labels are transformed into 3 mutually exclusive classes (Negative, Neutral, Positive).
4.  **Class Balancing**: The training set is balanced using the RandomOversampler technique to prevent model bias.
5.  **Model Training Gauntlet**: The balanced, vectorized data is used to train 6 different classifiers: Logistic Regression, Linear SVM, XGBoost, LightGBM, CatBoost, and Random Forest.
6.  **Final Evaluation**: All trained models are evaluated on the final, unseen test set to produce the definitive performance metrics.

## 🚀 Getting Started

To run this project yourself, follow these steps:

### 1\. Clone the Repository

```bash
git clone https://github.com/psywarrior1998/AffectiveLens.git
cd AffectiveLens
```

### 2\. Set Up the Environment

It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
```

  * Install the required packages from the notebook's setup cell.
  * Alternatively, create a `requirements.txt` file and run:

<!-- end list -->

```bash
pip install -r requirements.txt
```

### 3\. Access the Data

The notebook is configured to automatically download the pre-computed embeddings from the official Hugging Face Hub repository for this project: `psyrishi/MoodPulse`. No manual data download is required.

### 4\. Run the Notebook

Open the `AffectiveLens.ipynb` file in a Jupyter environment (like Jupyter Lab or VS Code) and execute the cells sequentially.

## 🏆 Results: Model Performance Leaderboard

The following table shows the final performance of all evaluated models on the unseen test set, based on the full execution of the notebook. The models are ranked by their F1-Score.

| Model | F1 Score (Micro) | Accuracy |
| :--- | :--- | :--- |
| 🥇 **LightGBM** | 0.6223 | 62.23% |
| 🥈 **XGBoost** | 0.6167 | 61.67% |
| 🥉 **Random Forest** | 0.6160 | 61.60% |
| **CatBoost** | 0.6108 | 61.08% |
| **Linear SVM** | 0.5973 | 59.73% |
| **Logistic Regression** | 0.5960 | 59.60% |

**Conclusion:**
The LightGBM model was the overall best performer, achieving the highest F1-Score and accuracy on the final test data. This indicates that for this specific high-dimensional embedding space, the gradient boosting algorithm was most effective at identifying the complex, non-linear boundaries between the emotion classes.

## 📦 Accessing Pre-Trained Models

All trained models from this study are saved and available on the Hugging Face Hub in the following repository: [psyrishi/affectivelens-emotion-models](https://huggingface.co/psyrishi/affectivelens-emotion-models).

You can easily download the champion model (LightGBM) or any other model using the `huggingface_hub` library.

```python
from huggingface_hub import hf_hub_download
import joblib

# Define the repository and the specific model file to download
REPO_ID = "psyrishi/affectivelens-emotion-models"
FILENAME = "LightGBM_MicroF1_0.6240.pkl" # Filename for the champion model

# Download the model file to a local path
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# Load the model
champion_model = joblib.load(model_path)
print("Successfully loaded the champion model:", champion_model)
```

## 🔬 Usage Example

The final deliverable is a simple function, `predict_emotion`, that can classify any new piece of text using the champion model.

### Example of how to use the final prediction function:

```python
input_text = "This is quite disappointing, I had hoped for a better outcome."

# The function returns the predicted label and its corresponding index
predicted_label, predicted_index = predict_emotion(input_text)

print(f'Text: "{input_text}"')
print(f'--> Predicted Emotion: {predicted_label} (index: {predicted_index})')

# Expected Output:
# Text: "This is quite disappointing, I had hoped for a better outcome."
# --> Predicted Emotion: negative (index: 0)
```

## 🔮 Future Work

  * **Full Model Fine-Tuning**: Instead of just using DistilBERT for feature extraction, the entire model could be fine-tuned on the GoEmotions dataset to potentially achieve higher performance.
  * **Advanced Sampling**: Explore more sophisticated techniques like SMOTE (Synthetic Minority Over-sampling Technique) to see if creating synthetic data points outperforms the current random oversampling method.
  * **Deep Error Analysis**: Conduct a thorough analysis of the champion model's misclassifications to identify patterns (e.g., struggles with sarcasm, irony) and guide future improvements.

## 🙏 Acknowledgments

This project utilizes the GoEmotions dataset, created by Google Research. The dataset is made available under the Creative Commons Attribution 4.0 International License. We thank the original authors for making this valuable resource publicly available for research.

## 📄 License

The code in this repository is licensed under the MIT License. See the LICENSE file for more details.
