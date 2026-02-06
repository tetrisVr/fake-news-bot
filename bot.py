#!/usr/bin/env python3
"""
🤖 Fake News Bot для Telegram
Простой детектор фейковых новостей
"""

import os
import sys
import asyncio
from threading import Thread
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import logging
from flask import Flask

print("=" * 50)
print("🤖 ЗАПУСК FAKE NEWS DETECTOR BOT")
print("=" * 50)

# 🔑 ВСТАВЬТЕ ВАШ ТОКЕН СЮДА!
BOT_TOKEN = "ВАШ_ТОКЕН_ЗДЕСЬ"  # ← ЗАМЕНИТЕ ЭТО!

# ========== 1. ОБУЧАЕМ МОДЕЛЬ ==========
print("🧠 Обучаю ИИ модель...")

data = {
    'text': [
        'Ученые Гарварда доказали вред молока',
        'Эксперты говорят о скрытой опасности',
        'Врачи Минздрава одобрили лекарство',
        'Источники сообщают о повышении пенсий',
        'Мэрия утвердит проект 15 марта',
        'Специалисты предупреждают о магнитных бурях',
        'Роспотребнадзор проверил 20 кафе',
        'Анонимные инсайдеры раскрыли тайну',
        'Исследование Оксфорда опубликовано в Nature',
        'Неизвестные ученые сделали открытие',
        'Школа №5 выиграла грант 100000 рублей',
        'Некоторые аналитики прогнозируют кризис',
        'Губернатор подписал указ 25 декабря',
        'Очевидцы утверждают о странных явлениях',
        'ВУЗ получил 5 новых лабораторий',
        'Отдельные политики требуют изменений',
        'Больница закупила 3 аппарата МРТ',
        'Инсайдеры слили секретные документы',
        'Компания Google инвестировала в проект',
        'Анонимный блогер раскрыл правду'
    ],
    'has_reliable_source': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'has_vague_source': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'has_specific_names': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    'has_concrete_dates': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'uses_absolute_words': [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    'has_urgent_call': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'contains_numbers': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
features = ['has_reliable_source', 'has_vague_source', 'has_specific_names',
            'has_concrete_dates', 'uses_absolute_words', 'has_urgent_call', 'contains_numbers']
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ========== 2. ФУНКЦИЯ ПРОВЕРКИ ==========
def check_news(news_text):
    text_lower = news_text.lower()
    
    reliable_keywords = ['минздрав', 'роспотребнадзор', 'губернатор', 'мэрия', 
                        'оксфорд', 'гарвард', 'университет', 'исследование',
                        'вуз', 'больница', 'врачи', 'ученые', 'компания']
    
    vague_keywords = ['эксперты', 'источники', 'специалисты', 'аналитики',
                     'очевидцы', 'инсайдеры', 'блогер', 'неизвестные',
                     'анонимные', 'отдельные', 'некоторые', 'люди']
    
    name_keywords = ['гарвард', 'минздрав', 'роспотребнадзор', 'оксфорд',
                    'губернатор', 'мэрия', '№', 'google', 'nature']
    
    date_keywords = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
    
    absolute_keywords = ['доказали', 'точно', 'абсолютно', 'несомненно', 
                        'утверждают', 'требуют', 'раскрыли', 'правда']
    
    urgent_keywords = ['срочно', 'немедленно', 'тревога', 'опасность', 
                      'предупреждают', 'важно', 'надо', 'нужно']
    
    has_reliable = 1 if any(word in text_lower for word in reliable_keywords) else 0
    has_vague = 1 if any(word in text_lower for word in vague_keywords) else 0
    has_names = 1 if any(word in text_lower for word in name_keywords) else 0
    has_dates = 1 if any(word in text_lower for word in date_keywords) else 0
    has_absolute = 1 if any(word in text_lower for word in absolute_keywords) else 0
    has_urgent = 1 if any(word in text_lower for word in urgent_keywords) else 0
    has_numbers = 1 if any(char.isdigit() for char in news_text) else 0
    
    features_array = [has_reliable, has_vague, has_names, has_dates, 
                     has_absolute, has_urgent, has_numbers]
    
    prediction = model.predict([features_array])[0]
    probability = model.predict_proba([features_array])[0]
    
    return {
        'text': news_text,
        'is_fake': bool(prediction),
        'fake_prob': float(probability[1] * 100),
        'true_prob': float(probability[0] * 100),
        'features': {
            'Надежный источник': has_reliable,
            'Размытый источник': has_vague,
            'Конкретные имена': has_names,
            'Конкретные даты': has_dates,
            'Абсолютные утверждения': has_absolute,
            'Срочный призыв': has_urgent,
            'Содержит цифры': has_numbers
        }
    }

# ========== 3. ВЕБ-СЕРВЕР ==========
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Fake News Bot</title>
        <style>
            body { font-family: Arial; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { background: white; padding: 30px; border-radius: 15px; max-width: 600px; margin: 0 auto; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
            h1 { color: #333; }
            .status { color: green; font-weight: bold; }
            .telegram-btn { background: #0088cc; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; display: inline-block; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Fake News Detector Bot</h1>
            <p class="status">✅ Бот активен</p>
            <p>Точность модели: {:.1%}</p>
            <p>Запущен: {}</p>
            <a href="https://t.me/fakenews_checker_bot" class="telegram-btn" target="_blank">
                🚀 Открыть в Telegram
            </a>
        </div>
    </html>
    """.format(accuracy, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def run_web():
    app.run(host='0.0.0.0', port=8080)

# ========== 4. TELEGRAM БОТ ==========
print("🤖 Создаю Telegram бота...")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔍 Проверить новость", callback_data='check')],
        [InlineKeyboardButton("📚 Примеры", callback_data='examples')],
        [InlineKeyboardButton("ℹ️ О боте", callback_data='about')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🤖 *ИИ Детектор Фейков*\n\n"
        f"Точность модели: {accuracy:.1%}\n"
        f"Отправь новость для проверки!",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text.startswith('/'):
        return
    
    news_text = update.message.text.strip()
    
    if len(news_text) < 10:
        await update.message.reply_text("❌ Слишком коротко. Минимум 10 символов.")
        return
    
    await update.message.reply_text("🔍 Анализирую...")
    
    result = check_news(news_text)
    
    features_text = ""
    for name, value in result['features'].items():
        symbol = "✅" if value == 1 else "❌"
        features_text += f"{symbol} {name}\n"
    
    if result['is_fake']:
        verdict = "🚨 *ВОЗМОЖНЫЙ ФЕЙК*"
    else:
        verdict = "✅ *ВЕРОЯТНО ПРАВДА*"
    
    response = (
        f"{verdict}\n\n"
        f"📊 *Статистика:*\n"
        f"Вероятность фейка: {result['fake_prob']:.1f}%\n"
        f"Вероятность правды: {result['true_prob']:.1f}%\n\n"
        f"🔎 *Признаки:*\n{features_text}"
    )
    
    await update.message.reply_text(response, parse_mode='Markdown')

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'check':
        await query.edit_message_text("📝 *Отправь новость для проверки*", parse_mode='Markdown')
    elif query.data == 'examples':
        text = ("🧪 *Примеры:*\n\n"
                "✅ Правда:\n`Минздрав одобрил вакцину 15 марта`\n\n"
                "🚨 Фейк:\n`Эксперты говорят о скрытом кризисе`")
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == 'about':
        text = ("🤖 *ИИ Детектор Фейков*\n\n"
                "Проверяет новости по 7 признакам:\n"
                "• Надежный источник\n• Конкретные имена\n• Цифры и даты\n\n"
                f"Точность: {accuracy:.1%}\n\n"
                "⚙️ Разработано для обучения ИИ")
        await query.edit_message_text(text, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")

# ========== 5. ЗАПУСК ==========
def main():
    print(f"✅ Модель обучена. Точность: {accuracy:.1%}")
    
    # Запускаем веб-сервер в отдельном потоке
    web_thread = Thread(target=run_web, daemon=True)
    web_thread.start()
    print("🌐 Веб-сервер запущен на порту 8080")
    
    # Проверяем токен
    if BOT_TOKEN == "ВАШ_ТОКЕН_ЗДЕСЬ":
        print("\n⚠️  ВНИМАНИЕ: Токен не установлен!")
        print("1. Получите токен у @BotFather в Telegram")
        print("2. Замените строку BOT_TOKEN в коде")
        print("3. Перезапустите бота")
        return
    
    # Создаем бота
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_error_handler(error_handler)
    
    print("🤖 Бот запускается...")
    print("📱 Откройте Telegram и напишите /start вашему боту")
    print("🔄 Для остановки: Ctrl+C")
    
    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()
