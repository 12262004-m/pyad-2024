import logging
import requests
import asyncio
import torch
import re
from collections import Counter
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from pymorphy2 import MorphAnalyzer
from g4f.client import Client


TOKEN = "8098528206:AAGGaTbY4kz3f9FYPf-2LP5EE4c2D1SWLKY"

bot = Bot(token=TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="📊 Получить анализ")]],
    resize_keyboard=True
)

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
morph = MorphAnalyzer()

STOP_WORDS = {
    "очень", "только", "если", "есть", "быть", "просто", "был", "так", "это", "после", "через", "нам", "его", "она",
    "также", "ещё", "всё", "самый", "хорошо", "ещё", "раз", "ну", "пару", "можно", "могут", "каждый", "какой", "какая",
    "лишь", "должен", "могу", "мочь", "более", "менее", "где", "куда", "когда", "снова", "именно", "всегда", "никогда",
    "уже", "теперь", "совсем", "почти", "нужно", "лучше", "вообще", "конечно", "пожалуйста", "спасибо", "один", "два",
}

client = Client()


def clean_text(text):
    text = re.sub(r'[^\w\s,.!?ёЁа-яА-Я]', '', text).strip().lower()
    return text


def get_reviews(url):
    response = requests.get(url)
    if response.status_code != 200:
        return ["Ошибка при загрузке данных. Проверьте ссылку."]

    soup = BeautifulSoup(response.text, "html.parser")
    reviews = [clean_text(r.text) for r in soup.findAll('span', class_='business-review-view__body-text')]
    return reviews if reviews else ["На данной странице нет отзывов."]


def normalize_word(word):
    return morph.parse(word)[0].normal_form


def delete_stop_words(text):
    words = [normalize_word(word) for word in text.split() if word not in STOP_WORDS and len(word) > 3]
    return words


def analyze_sentiment(reviews):
    results = {"negative": 0, "neutral": 0, "positive": 0}
    details = []
    keywords = {"negative": [], "positive": []}

    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probabilities).item()
        sentiment_label = ["negative", "neutral", "positive"][sentiment]
        results[sentiment_label] += 1
        details.append({"text": review, "sentiment": sentiment_label, "probabilities": probabilities.tolist()})
        if sentiment_label in ["negative", "positive"]:
            keywords[sentiment_label].extend(delete_stop_words(review))
    return results, details, keywords


def save_analysis_to_file(details):
    file_path = "sentiment_analysis_results.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(f"Отзыв: {d['text']}\n")
            f.write(f"Тональность: {d['sentiment']}\n")
            f.write(f"Вероятности: {d['probabilities']}\n\n")
    return file_path


def generate_recommendation_with_gpt(common_words):
    if not common_words:
        return "Нет конкретных рекомендаций."

    prompt = (f"Ты - эксперт по ресторанному бизнесу. Дай краткую рекомендацию по улучшению сервиса заведения общественного питания на "
              f"основе отзывов клиентов. Упомяни ключевые слова, а именно проблемные места, на которые "
              f"чаще всего жалуются клиенты: {', '.join(common_words)}. "
              f"Рекомендация должна быть полезной, но не слишком длинной. Твоя цель - обратить внимание на "
              f"проблемные точки, рассказать о них кратко и понятно, а также предложить вариант улучшения"
              f"для избавления от проблемных мест. Напиши только КРАТКИЙ ответ в НЕСКОЛЬКО ПРЕДЛОЖЕНИЙ! НЕ ИСПОЛЬЗУЙ"
              f"ЭМОДЖИ И ЖИРНЫЙ ШРИФТ!")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        web_search=False
    )
    return response.choices[0].message.content


def summary_reviews_with_gpt(common_words):
    if not common_words:
        return "Нет конкретных рекомендаций."

    prompt = (f"Ты - эксперт по ресторанному бизнесу. Дай краткий отчет на основе ключевых слов, "
              f"а именно тех вещей, о которых люди положительно откликались чаще всего при посещении заведения: "
              f"{', '.join(common_words)}. Отчет должен быть полезным и кратким, твоя задача - обобщить хорошие моменты "
              f"и датоь максимально краткую обратную связь с рекомендацией по доп улучшениям. Напиши только КРАТКИЙ "
              f"ответ в НЕСКОЛЬКО ПРЕДЛОЖЕНИЙ! НЕ ИСПОЛЬЗУЙ ЭМОДЖИ И ЖИРНЫЙ ШРИФТ!")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        web_search=False
    )
    return response.choices[0].message.content


def generate_recommendations(keywords):
    recommendations = {}
    for sentiment, words in keywords.items():
        common_words = [word for word, _ in Counter(words).most_common(5)]
        if sentiment == "negative":
            recommendations[sentiment] = {
                "words": common_words,
                "recommendation": generate_recommendation_with_gpt(common_words)
            }
        else:
            recommendations[sentiment] = {
                "words": common_words,
                "recommendation": summary_reviews_with_gpt(common_words)
            }
    return recommendations


def visualize_distribution(results):
    plt.figure(figsize=(8, 6))
    plt.bar(results.keys(), results.values(), color=['red', 'yellow', 'green'])
    plt.title('Распределение тональности отзывов')
    plt.xlabel('Категории тональности')
    plt.ylabel('Количество отзывов')
    plt.xticks(["negative", "neutral", "positive"], ["Негатив", "Нейтрально", "Позитив"])
    plt.grid(axis='y', alpha=0.7)
    plt.savefig("sentiment_distribution.png")
    plt.close()


@dp.message(F.text == "/start")
async def send_welcome(message: Message):
    await message.answer(
        "👋 Привет! Я анализирую отзывы с Яндекс.Карт и даю грамотные рекомендации. \n"
        "Чтобы начать анализ, нажмите на кнопку ниже",
        reply_markup=keyboard
    )


@dp.message(F.text == "📊 Получить анализ")
async def ask_for_link(message: Message):
    await message.answer("Отправьте ссылку на заведение в Яндекс.Картах.")


@dp.message()
async def process_reviews(message: Message):
    url = message.text.strip()
    if "yandex.ru/maps" not in url:
        await message.reply("❌ Ошибка в URL. Данная ссылка не соответствует Яндекс.Картам.")
        return
    await message.reply("⏳ Собираю и анализирую отзывы...")
    reviews = get_reviews(url)
    if len(reviews) == 1 and "Ошибка" in reviews[0]:
        await message.reply(reviews[0])
        return
    results, details, keywords = analyze_sentiment(reviews)
    recommendations = generate_recommendations(keywords)
    visualize_distribution(results)

    rec_text = "\n\n".join([
        f"➡️ {s.upper()}:\nСлова: {', '.join(recommendations[s]['words'])}\n"
        f"{recommendations[s]['recommendation']}"
        for s in recommendations
    ])

    summary = (f"Анализ завершён!\n"
               f"🔴 Негативных: {results['negative']}\n"
               f"🟡 Нейтральных: {results['neutral']}\n"
               f"🟢 Позитивных: {results['positive']}\n\n\n\n"
               f"{rec_text}")

    await message.answer_photo(FSInputFile("sentiment_distribution.png"), caption="📊 График распределения отзывов")
    await message.answer(summary)
    report_file = save_analysis_to_file(details)
    await message.answer_document(FSInputFile(report_file), caption="Подробный анализ отзывов")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
