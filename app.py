from flask import Flask, request, render_template
import joblib
from transformers import pipeline

# --- (optional): your response generator via Transformers ---
# If you want to continue offering automatic responses for non-spam emails:
generator = pipeline("text2text-generation", model="google/flan-t5-base")
# or, if you prefer DialoGPT:
# generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")


def generate_response(mail):
    prompt = (
        "You are a professional assistant.\n"
        f"The user received the following email:\n\n\"{mail}\"\n\n"
        "Write a polite and relevant reply to this message.\n"
        "Respond directly to what is written, and do not invent any context.\n"
        "If the email is just a reminder or a confirmation, acknowledge it and confirm politely."
    )

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1
    )[0]["generated_text"]

    return output.strip()



app = Flask(__name__)

# --- Loading the complete pipeline (vectorizer + LogisticRegression) ---
# The file at the root of your Flask project should be:
# models/logreg_best_model.joblib
model = joblib.load("models/logreg_best_model.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestion = None

    if request.method == "POST":
        email = request.form["email"].strip()
        if not email:
            result = "🚫 Mail cannot be empty."
        else:
            # Since the pipeline already includes TfidfVectorizer, we can predict directly from raw text
            pred = model.predict([email])[0]
            # Pipeline returns 0 (Safe Email) or 1 (Phishing Email)
            if pred == 1:
                result = "🚫 This e-mail is SPAM. Delete it immediately."
            else:
                result = "✅ This e-mail is NOT spam."
                # Optional: generate automatic response suggestion if not spam
                suggestion = generate_response(email)
                print(suggestion)

    return render_template("index.html", result=result, suggestion=suggestion)


if __name__ == "__main__":
    app.run(debug=True)
