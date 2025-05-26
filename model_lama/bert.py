import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class BERTChatbot:
    def __init__(self, dataset_path, model_name='sentence-transformers/all-MiniLM-L6-v2', top_n=1, mode='simple'):
        self.model = SentenceTransformer(model_name)
        self.dataset_path = dataset_path
        self.top_n = top_n
        self.mode = mode

        # Inisialisasi stemmer Bahasa Indonesia
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        self.dataset = self.load_dataset()
        self.questions = [self.preprocess(item['pertanyaan']) for item in self.dataset]
        self.answers = [item['jawaban'] for item in self.dataset]
        self.codes = [item['kode'] for item in self.dataset]

        self.embeddings_path = dataset_path.replace(".json", "_embeddings.npy")
        self.embeddings = self.load_or_create_embeddings()

    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def preprocess(self, text):
        # Lowercase & bersihkan karakter non-alfanumerik
        text = re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()
        # Tokenisasi
        tokens = text.split()
        # Stemming per token
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        # Gabungkan kembali untuk dipakai oleh encoder
        return ' '.join(stemmed_tokens)

    def load_or_create_embeddings(self):
        if os.path.exists(self.embeddings_path):
            return np.load(self.embeddings_path)
        else:
            embeddings = self.model.encode(self.questions)
            np.save(self.embeddings_path, embeddings)
            return embeddings

    def get_answer(self, user_input, threshold=0.75):
        if len(user_input.strip()) < 3:
            return "Pertanyaan terlalu singkat. Bisa dijelaskan lebih detail?", None

        clean_input = self.preprocess(user_input)
        user_embedding = self.model.encode([clean_input], convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, self.embeddings)[0].cpu().numpy()

        if self.mode == 'simple':
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            if best_score < threshold:
                return "Maaf, saya tidak mengerti pertanyaan tersebut.", None
            return self.answers[best_idx], self.codes[best_idx]

        elif self.mode == 'top-n':
            return self.get_top_n_answers(scores, threshold)

    def get_top_n_answers(self, scores, threshold):
        top_indices = scores.argsort()[-self.top_n:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] >= threshold:
                results.append({
                    "skor": float(scores[idx]),
                    "jawaban": self.answers[idx],
                    "kode": self.codes[idx]
                })
        if not results:
            return [{"jawaban": "Maaf, saya tidak mengerti pertanyaan tersebut.", "kode": None}]
        return results
