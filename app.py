from flask import Flask, request, render_template
import joblib
from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def generate_response(mail):
    prompt = (
        f"L'utilisateur a reçu ce mail : \"{mail}\"\n"
        f"Voici la meilleure réponse possible :"
    )
    output = generator(prompt, max_length=250, num_return_sequences=1)[0]["generated_text"]
    print(output)
    # On isole la réponse générée après le prompt
    return output.split("meilleure réponse possible :")[-1].strip()

app = Flask(__name__)
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestion = None
    if request.method == "POST":
        email = request.form["email"]
        if(len(email.strip()) == 0):
            result = "🚫 Mail cannot be empty." 
        else:
            X = vectorizer.transform([email])
            pred = model.predict(X)[0]

            if pred == "spam":
                result = "🚫 This e-mail is SPAM. Delete it immediately."
            elif pred == "inutile":
                result = "ℹ️ This e-mail seems unnecessary."
            elif pred == "utile":
                result = "✅ This e-mail is useful."
                suggestion = "Suggested answer: Hello, thank you for your message. I'll get back to you shortly."
                suggestion = generate_response(email)

    return render_template("index.html", result=result, suggestion=suggestion)

if __name__ == "__main__":
    app.run(debug=True)
