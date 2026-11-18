"""Deterministic, openly synthetic Ukrainian IT corpus."""

from pathlib import Path

TOPICS = [
    "Python підтримує генератори ітератори та обробку даних. Програма читає текст і створює індекс для пошуку.",
    "База даних зберігає записи. SQL виконує запити та оптимізує вибірку. Індекс прискорює пошук документів.",
    "Тестування програм перевіряє функції та помилки. Автоматичні тести допомагають підтримувати якість коду.",
    "Мережевий сервер обробляє запити клієнтів. Протокол передає повідомлення та підтримує захист даних.",
    "Машинне навчання аналізує тексти. Модель перетворює слова на ознаки для класифікації документів.",
    "Алгоритм сортує дані та вимірює складність. Пам'ять і швидкість впливають на продуктивність програми.",
    "Вебзастосунок використовує API та сервер. Користувач надсилає запит і отримує результат пошуку.",
    "Система контролю версій Git зберігає зміни коду. Команда розробляє проєкт і перевіряє тести.",
    "Unicode забезпечує обробку української мови. Апострофи, цифри 2026 та дефіси важливі для токенізації.",
    "Пошукова система знаходить документи за словами. Корпус текстів є основою для побудови індексу.",
]


def generate_corpus(directory: Path | str, count: int = 1000, repetitions: int = 20) -> None:
    if count < 1 or repetitions < 1:
        raise ValueError("count and repetitions must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        topic = TOPICS[index % len(TOPICS)]
        # Predictable variation, no randomness, no copyrighted downloaded text.
        text = (f"Документ {index + 1}. Тема: інформаційні технології.\n" + topic + "\n") * repetitions
        (directory / f"doc_{index + 1:05d}.txt").write_text(text, encoding="utf-8")
