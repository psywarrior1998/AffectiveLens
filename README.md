**Title:**
**AffectiveLens: A Transformer-Based Engine for Emotion Detection**

**Overview:**
AffectiveLens is an end-to-end AI project that utilizes state-of-the-art DistilBERT Transformer embeddings to detect emotions in text. It evaluates six machine learning models to classify text sentiment into positive, negative, or neutral. This system demonstrates a comprehensive NLP pipeline for emotion detection.

**Key Features:**

* **State-of-the-Art Embeddings**: Leverages the **distilbert-base-uncased** model for high-quality contextual embeddings.
* **Comprehensive Model Comparison**: Benchmarks six machine learning models to find the best performer.
* **Robust Data Processing**: Complete pipeline for cleaning, balancing (oversampling), and preparing the data.
* **End-to-End Workflow**: From data ingestion to deployment of a final, deployable prediction function.
* **Reproducible Environment**: Ensures consistent and reproducible results through a version-pinned environment.

**Technologies & Tools:**

* **DistilBERT** (Hugging Face)
* **Python** (Jupyter Notebook, scikit-learn, XGBoost, LightGBM, CatBoost, Random Forest)
* **Hugging Face Hub** (for pre-computed embeddings)
* **Joblib** (for model serialization)

**Project Workflow:**

1. **Environment Setup**: Creates a version-controlled environment to ensure reproducibility.
2. **Data Ingestion**: Downloads pre-computed embeddings from Hugging Face.
3. **Data Preparation**: Transforms the dataset into three emotion categories (Negative, Neutral, Positive).
4. **Class Balancing**: Balances the data using Random Oversampling to avoid bias.
5. **Model Training**: Trains six classifiers: Logistic Regression, SVM, XGBoost, LightGBM, CatBoost, Random Forest.
6. **Final Evaluation**: Models are evaluated on an unseen test set based on F1-Score and accuracy.

**Getting Started:**

1. **Clone the repository**

   ```bash
   git clone https://github.com/psywarrior1998/AffectiveLens.git
   cd AffectiveLens
   ```
2. **Set up the environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Access the Data**
   The data is automatically fetched from the Hugging Face repository.
4. **Run the Notebook**
   Execute the `AffectiveLens.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, VS Code).

**Model Performance Leaderboard:**

| Model                   | F1 Score (Micro) | Accuracy |
| ----------------------- | ---------------- | -------- |
| 🥇 **LightGBM**         | 0.6223           | 62.23%   |
| 🥈 **XGBoost**          | 0.6167           | 61.67%   |
| 🥉 **Random Forest**    | 0.6160           | 61.60%   |
| **CatBoost**            | 0.6108           | 61.08%   |
| **Linear SVM**          | 0.5973           | 59.73%   |
| **Logistic Regression** | 0.5960           | 59.60%   |

**Champion Model:**

* **LightGBM** achieved the highest F1-Score and accuracy, making it the best performer for this task.

**Accessing Pre-Trained Models:**
Trained models, including the champion **LightGBM**, are available on [Hugging Face](https://huggingface.co/psyrishi/affectivelens-emotion-models).

Example code to download and use the model:

```python
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "psyrishi/affectivelens-emotion-models"
FILENAME = "LightGBM_MicroF1_0.6240.pkl"

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
champion_model = joblib.load(model_path)
print("Successfully loaded the champion model:", champion_model)
```

**Usage Example:**
Using the `predict_emotion` function for classifying text:

```python
input_text = "This is quite disappointing, I had hoped for a better outcome."
predicted_label, predicted_index = predict_emotion(input_text)

print(f'Text: "{input_text}"')
print(f'--> Predicted Emotion: {predicted_label} (index: {predicted_index})')
```

**Future Work:**

* **Fine-Tuning**: Fine-tune the entire DistilBERT model for better performance.
* **Advanced Sampling**: Experiment with SMOTE (Synthetic Minority Over-sampling Technique) for better class balance.
* **Error Analysis**: Conduct a deeper analysis of misclassifications (e.g., sarcasm or irony).

**Acknowledgments:**
This project utilizes the **GoEmotions dataset** by Google Research under the Creative Commons Attribution 4.0 International License.

**License:**
MIT License (see LICENSE file for details).

