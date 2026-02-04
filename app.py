from flask import Flask, request, jsonify
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# validate config at startup
Config.validate()


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if len(question) > app.config["MAX_QUESTION_LENGTH"]:
        return jsonify({"error": "Question too long"}), 400

    # placeholder for RAG logic
    return jsonify({
        "answer": "Config system works.",
        "question": question,
        "top_k": app.config["TOP_K"]
    })
