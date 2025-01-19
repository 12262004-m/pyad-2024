import pandas as pd
from torch.nn.functional import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def analyze_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probabilities).item()
        results.append({
            "text": text,
            "sentiment": sentiment,
            "probabilities": probabilities.detach().numpy()
        })
    return results

def load_reviews(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    return [review.strip() for review in reviews if review.strip() and review.strip() != "---------"]

def visualize_distribution(distribution):
    categories = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=['red', 'yellow', 'green'])
    plt.title('Распределение тональности отзывов')
    plt.xlabel('Категории тональности')
    plt.ylabel('Количество отзывов')
    plt.xticks(categories, labels=["Негатив", "Нейтрально", "Позитив"])
    plt.grid(axis='y', alpha=0.7)
    plt.show()

def main():
    file_path = "data.txt"
    reviews = load_reviews(file_path)
    sentiment_results = analyze_sentiment(reviews)

    distribution = {"negative": 0, "neutral": 0, "positive": 0}
    for result in sentiment_results:
        if result["sentiment"] == 0:
            distribution["negative"] += 1
        elif result["sentiment"] == 1:
            distribution["neutral"] += 1
        elif result["sentiment"] == 2:
            distribution["positive"] += 1

    print("Распределение тональности:")
    for sentiment, count in distribution.items():
        print(f"{sentiment}: {count}")

    with open("sentiment_analysis_results.txt", "w", encoding="utf-8") as f:
        for result in sentiment_results:
            f.write(f"Отзыв: {result['text']}\n")
            f.write(f"Тональность: {['негатив', 'нейтрально', 'позитив'][result['sentiment']]}\n")
            f.write(f"Вероятности: {result['probabilities']}\n")
            f.write("\n")
    visualize_distribution(distribution)


if __name__ == "__main__":
    main()
