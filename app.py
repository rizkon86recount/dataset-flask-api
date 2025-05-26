from flask import Flask, request, jsonify
from bert import BERTChatbot

app = Flask(__name__)

# Inisialisasi model chatbot
chatbot = BERTChatbot("./DATASET_PEMROGRAMAN/data.json", mode='simple')
# chatbot = BERTChatbot("./DATASET_PEMROGRAMAN/dataset.json", mode="top-n", top_n=3)
# print(chatbot.preprocess("Bagaimana penulisan JavaScript secara internal?"))


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("pertanyaan")

    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

    jawaban, kode = chatbot.get_answer(question)

    return jsonify({
        "jawaban": jawaban,
        "kode": kode
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)