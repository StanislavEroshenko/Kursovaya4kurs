import os
import re
import pandas as pd
import joblib
from tkinter import Tk, Text, Button, Label, filedialog, END, messagebox, Scrollbar, RIGHT, Y
from sklearn.linear_model import LogisticRegression

def extract_features(text: str):
    words = re.findall(r'\b\w+\b', text)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    punctuation_count = len(re.findall(r'[.,!?;:]', text))
    punctuation_per_100_words = (punctuation_count / len(words)) * 100 if words else 0

    adverbs_conjunctions = re.findall(r'\b(и|но|или|однако|когда|где|как|очень|всегда)\b', text, re.IGNORECASE)
    adverb_conjunction_freq = len(adverbs_conjunctions) / len(words) if words else 0

    has_structures = int(bool(re.search(r'[-*•]\s|\n\d+\.', text)))

    return [
        avg_word_length,
        avg_sentence_length,
        punctuation_per_100_words,
        adverb_conjunction_freq,
        has_structures
    ]

def train_model():
    data = [
        {"text": "Это простой текст с короткими предложениями.", "complexity": 0},
        {"text": "Этот текст имеет умеренно сложную структуру и некоторые слова длиннее.", "complexity": 1},
        {"text": "Сложные технические документы содержат специализированную лексику и длинные предложения.", "complexity": 2},
        {"text": "Кот сидел на ковре.", "complexity": 0},
        {"text": "Руководство пользователя содержит шаги и подзаголовки.", "complexity": 1},
        {"text": "Научные статьи включают таблицы, списки и подробные объяснения.", "complexity": 2}
    ]

    df = pd.DataFrame(data)
    X = pd.DataFrame([extract_features(text) for text in df['text']])
    y = df['complexity']

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, 'text_complexity_model.joblib')
    print("Модель обучена и сохранена.")

def generate_explanation(features, prediction):
    classes = ["простой", "средний", "сложный"]
    explanation = f"Результат: текст классифицирован как \"{classes[prediction]}\"\n\n"
    explanation += f"- Средняя длина слов: {features[0]:.2f}\n"
    explanation += f"- Средняя длина предложений: {features[1]:.2f}\n"
    explanation += f"- Пунктуация на 100 слов: {features[2]:.2f}\n"
    explanation += f"- Наречия и союзы: {features[3]:.2f}\n"
    explanation += f"- Структурированные элементы: {'да' if features[4] else 'нет'}"
    return explanation

def load_text_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if path:
        with open(path, encoding='utf-8') as f:
            content = f.read()
            text_input.delete(1.0, END)
            text_input.insert(END, content)

def classify_text():
    text = text_input.get("1.0", END).strip()
    if not text:
        messagebox.showwarning("Пустой ввод", "Введите или загрузите текст.")
        return

    if not os.path.exists("text_complexity_model.joblib"):
        train_model()

    model = joblib.load("text_complexity_model.joblib")
    features = extract_features(text)
    prediction = model.predict([features])[0]
    explanation = generate_explanation(features, prediction)

    output_label.config(text=explanation)

root = Tk()
root.title("Оценка сложности текста")

Label(root, text="Введите или загрузите текст:").pack()

scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

text_input = Text(root, height=10, width=80, wrap="word", yscrollcommand=scrollbar.set)
text_input.pack(padx=10, pady=5)
scrollbar.config(command=text_input.yview)

Button(root, text="Загрузить текст из файла", command=load_text_file).pack(pady=2)
Button(root, text="Оценить сложность", command=classify_text).pack(pady=5)

output_label = Label(root, text="", justify="left", wraplength=600, fg="darkblue")
output_label.pack(pady=10)

root.mainloop()