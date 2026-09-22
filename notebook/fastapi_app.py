import json
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# STEP 1: Load the saved model, tokenizer, and settings.

model = load_model(r"D:\class\main project\notebook\artifacts\next_word_model.keras")

with open(r"D:\class\main project\notebook\artifacts\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(r"D:\class\main project\notebook\artifacts\config.json", "r") as f:
    config = json.load(f)

MAX_SEQ_LEN = config["MAX_SEQ_LEN"]

# STEP 2: The prediction function.

def predict_next_words(seed_text, num_words=3):
    text = seed_text

    for _ in range(num_words):
        # Turn the text into numbers the model understands
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LEN - 1, padding="pre")

        # Ask the model: what's the most likely next word?
        predicted_id = np.argmax(model.predict(token_list, verbose=0))

        # Turn that number back into a real word
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                predicted_word = word
                break

        # Add the predicted word and repeat
        text = text + " " + predicted_word

    return text

# STEP 3: Define what a request looks like.

class PredictRequest(BaseModel):
    text: str
    num_words: int = 3

# STEP 4: Create the app and its two endpoints.

app = FastAPI(title="Next Word Prediction API")


@app.get("/")
def home():
    return {"message": "API is running. Go to /docs to try it out."}


@app.post("/predict")
def predict(request: PredictRequest):
    result = predict_next_words(request.text, request.num_words)
    return {
        "input": request.text,
        "prediction": result
    }
