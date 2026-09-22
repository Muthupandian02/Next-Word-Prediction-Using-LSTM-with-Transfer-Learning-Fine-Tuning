# Next-Word-Prediction-Using-LSTM-with-Transfer-Learning-Fine-Tuning

## 1. Project Overview

This project builds a **Next Word Prediction** system — given some starting text, the model predicts what word (or words) are likely to come next. It's trained specifically on cooking recipe instructions, so it learns the patterns of how recipe steps are typically written (e.g. "preheat the oven to" → "350 degrees", "mix the flour and" → "sugar").

The project has two parts:
1. A **training notebook** that downloads data, cleans it, trains a model, and evaluates it.
2. A **FastAPI web API** that loads the trained model and lets you get predictions over HTTP, without needing to touch the notebook again.

**Author:** Muthupandian S
**Background:** Biomedical Engineering, applying deep learning and NLP techniques to a text-generation problem.

---

## 2. How the Model Works (Plain-English Explanation)

At a high level:

1. We download ~2000 recipe instructions from a public dataset.
2. We clean the text (lowercase everything, remove punctuation/numbers, drop very short or duplicate recipes).
3. We break each recipe into overlapping mini-examples — e.g. "mix the flour and sugar" becomes:
   - "mix the" → "flour"
   - "mix the flour" → "and"
   - "mix the flour and" → "sugar"
4. We turn words into numbers using a **Tokenizer** (a lookup table of word → number), and pad everything to the same length.
5. We train an **LSTM (Long Short-Term Memory) neural network** — a type of model well-suited to sequences like text — to predict the next word given the previous words.
6. We compare three versions of this model (explained below) to see which works best.
7. We save the winning model so the API can use it without retraining.

### Why three models?

| Model | What it does | Why we build it |
|---|---|---|
| **1. Baseline** | Learns word meanings completely from scratch, using only our ~2000 recipes | Gives us an honest number to compare against — without this, we can't tell if GloVe actually helped |
| **2. Transfer Learning (frozen)** | Starts from GloVe embeddings — word vectors pretrained on billions of words — and keeps them **frozen** while only training the LSTM | Reuses general language knowledge the model couldn't otherwise learn from just 2000 recipes |
| **3. Fine-tuned** | Same as #2, but then **unfreezes** the embeddings and continues training with a very small learning rate | Lets GloVe's general knowledge adapt slightly to recipe-specific language, without erasing what it already knows |

We evaluate all three the same way, on the same untouched test set, so the comparison is fair.

### Why the train/test split happens *before* cleaning

In an earlier version of this project, the train/test split happened *after* the recipes had already been broken into overlapping mini-examples. That let pieces of the *same* recipe end up in both the training set and the test set — so the model could look better than it really was, because it had partially already seen that recipe. This is called **data leakage**.

To fix it, the raw recipes are split into train/test *first*, before any cleaning or tokenizing. Each recipe goes entirely into one side or the other. This means the test set is genuinely unseen data, and the final accuracy numbers can be trusted.

---

## 3. Files in This Project

```
main project/
├── notebook/
│   └── next_word_prediction_improved_v2.ipynb   # training notebook
├── fastapi_app.py                                # API server
├── artifacts/                                    # created after running the notebook
│   ├── next_word_model.keras                     # the trained model
│   ├── tokenizer.pkl                              # word ↔ number lookup table
│   └── config.json                                # small settings the API needs
└── README.md                                      # this file
```

| File | Purpose |
|---|---|
| `next_word_prediction_improved_v2.ipynb` | Everything from downloading data to saving the final trained model. Run this first. |
| `fastapi_app.py` | A small web server that loads the saved model and answers prediction requests. Run this after the notebook. |
| `artifacts/next_word_model.keras` | The trained neural network, saved in Keras's native format. |
| `artifacts/tokenizer.pkl` | The tokenizer (word-to-number mapping) used during training — needed so the API converts new text the same way. |
| `artifacts/config.json` | Stores settings like `MAX_SEQ_LEN` so the API pads text exactly the way the model expects. |

---

## 4. Requirements

- Python 3.9 or newer
- Internet access (only needed once, for training — to download the dataset and GloVe embeddings)

### Python packages

For the notebook:
```
pandas
numpy
requests
matplotlib
scikit-learn
tensorflow
gensim
```

For the API:
```
fastapi
uvicorn
tensorflow
numpy
```

Install everything with:
```
pip install pandas numpy requests matplotlib scikit-learn tensorflow gensim fastapi uvicorn
```

---

## 5. Step-by-Step: Training the Model

1. Open `notebook/next_word_prediction_improved_v2.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or Kaggle.
2. Make sure the environment has internet access — the notebook downloads:
   - ~2000 recipes from a Hugging Face dataset
   - Pretrained GloVe word embeddings (~100MB, via `gensim`)
3. Run every cell from top to bottom, in order. Roughly what happens:
   - **Steps 1–2:** Import libraries, download recipes.
   - **Step 3:** Split raw recipes into train/test (before any cleaning).
   - **Steps 4–7:** Clean text, tokenize, build training sequences, pad them.
   - **Step 8:** Load GloVe embeddings.
   - **Steps 9–12:** Train the baseline model, then the frozen transfer-learning model, then fine-tune it.
   - **Steps 13–15:** Compare all three models with a table, chart, and sample predictions.
   - **Step 16:** Save the final model, tokenizer, and config into `artifacts/`.
4. Training time depends on your machine — expect anywhere from a few minutes to 20–30 minutes on a typical laptop CPU, longer without a GPU.
5. When it's done, confirm the `artifacts/` folder was created and contains all three files listed above.

---

## 6. Step-by-Step: Running the API

1. Make sure `fastapi_app.py` sits in a folder where an `artifacts/` folder (from Step 16 of the notebook) is also present nearby — the app looks for `artifacts/` relative to its own file location, so this should work automatically once both are in the project folder.
2. Start the server:
   ```
   uvicorn fastapi_app:app --reload --port 8000
   ```
   or simply:
   ```
   python fastapi_app.py
   ```
3. You should see output like:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```
4. Leave this terminal running — the server stays active until you stop it (Ctrl+C).

---

## 7. Using the API

### Option A: Browser (easiest)
Go to:
```
http://127.0.0.1:8000/docs
```
This opens an interactive Swagger UI where you can type in text and click "Execute" to get a prediction — no coding required.

### Option B: curl
```
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "place fried chicken", "num_words": 3}'
```

### Option C: Python
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "preheat the oven to", "num_words": 2}
)
print(response.json())
```

### Request format
| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | The seed text to continue |
| `num_words` | integer | No (default: 3) | How many words to predict |

### Example response
```json
{
  "input": "place fried chicken",
  "prediction": "place fried chicken until golden brown"
}
```

### Health check
```
GET http://127.0.0.1:8000/
```
Returns a simple message confirming the API is running.

---

## 8. Model Performance

The notebook evaluates all three models (baseline, frozen transfer learning, fine-tuned) on the same held-out test set and reports:
- **Accuracy** — how often the model's top prediction is exactly right
- **Top-3 Accuracy** — how often the correct word is in the model's top 3 guesses
- **Precision** and **Recall** (weighted)
- **Loss**

Exact numbers vary slightly between runs (random initialization, dataset refresh, etc.), but in general the fine-tuned model should outperform the frozen-embedding model, which should outperform the from-scratch baseline — see the comparison table and chart generated in Step 14 of the notebook for the actual numbers from your run.

---

## 9. Known Limitations

- The model only knows recipe-style language — feeding it unrelated text (e.g. news headlines) will likely produce nonsense predictions.
- Vocabulary is capped at 5000 words; rare or unusual words are treated as `<OOV>` (out-of-vocabulary) and won't be predicted directly.
- The model looks at a maximum of 14 previous words (`MAX_SEQ_LEN - 1`) — anything earlier is truncated.
- No spelling correction or grammar checking is applied to predictions.

## 10. Possible Future Improvements

- Train on a larger and more diverse recipe dataset for broader vocabulary coverage.
- Try a Bidirectional LSTM or a Transformer-based architecture for potentially better accuracy.
- Add beam search instead of always picking the single most likely word.
- Deploy the API with Docker for easier hosting.
- Add a simple frontend so non-technical users can try it without curl or `/docs`.
