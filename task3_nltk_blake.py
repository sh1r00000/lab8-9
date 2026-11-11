"""Практична робота №8–9, варіант 5. Завдання 3–6: NLTK, Blake poems."""
from pathlib import Path
import re
import argparse
import json
import nltk
import requests
from nltk import FreqDist
from nltk.stem import PorterStemmer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
OUT = BASE / 'output'
CHARTS = BASE / 'charts'
FILE_ID = 'blake-poems.txt'


def load_blake(demo=False):
    if demo:
        return (DATA / 'blake_excerpt_for_test.txt').read_text(encoding='utf-8'), True
    # Якщо текст локально збережений, додаткове завантаження не потрібне.
    local = DATA / FILE_ID
    if local.exists():
        return local.read_text(encoding='utf-8'), False
    try:
        from nltk.corpus import gutenberg
        return gutenberg.raw(FILE_ID), False
    except LookupError:
        print('Корпус Gutenberg відсутній, спроба встановлення засобами NLTK...')
        if nltk.download('gutenberg', quiet=True, raise_on_error=False):
            from nltk.corpus import gutenberg
            return gutenberg.raw(FILE_ID), False
        mirrors = [
            'https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/gutenberg/blake-poems.txt',
            'https://research.cs.wisc.edu/zhu/s/TTP/nlp/nltk/data/corpora/gutenberg/blake-poems.txt',
        ]
        for url in mirrors:
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                text = response.content.decode('utf-8-sig')
                if 'SONGS OF INNOCENCE' in text and len(text) > 30000:
                    local.write_text(text, encoding='utf-8')
                    return text, False
            except (requests.RequestException, UnicodeError, OSError):
                continue
        raise RuntimeError('Не вдалося завантажити Gutenberg. Підключіть інтернет і виконайте '
                           '`python -m nltk.downloader gutenberg`, або покладіть blake-poems.txt '
                           'у папку data/. Для тесту без інтернету: --demo')


def frequencies(tokens):
    # Сортування за частотою, а за рівності — за абеткою.
    return sorted(FreqDist(tokens).items(), key=lambda item: (-item[1], item[0]))[:10]


def chart(top, filename, title):
    words, numbers = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.bar(words, numbers)
    ax.set_title(title)
    ax.set_ylabel('Частота')
    ax.tick_params(axis='x', labelrotation=35)
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(CHARTS / filename, dpi=170)
    plt.close(fig)


def analyze(demo=False):
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    raw_text, is_demo = load_blake(demo)
    # Виділяємо англійські слова; пробіли та розділові знаки не стають словами.
    tokens = [word.lower() for word in re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", raw_text)]
    if not tokens:
        raise ValueError('У тексті немає слів для аналізу')
    stopwords = set((DATA / 'english_stopwords.txt').read_text(encoding='utf-8').splitlines())
    cleaned = [w for w in tokens if w not in stopwords]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in cleaned]
    raw_top, clean_top, stem_top = frequencies(tokens), frequencies(cleaned), frequencies(stems)
    title = 'ФРАГМЕНТ ДЛЯ ПЕРЕВІРКИ' if is_demo else 'ПОВНИЙ ТЕКСТ BLAKE-POEMS.TXT'
    print('\n=== ЗАВДАННЯ 3–6. АНАЛІЗ ТЕКСТУ NLTK ===')
    print('Режим:', title)
    print('Загальна кількість слів:', len(tokens))
    print('Унікальних слів:', len(set(tokens)))
    print('Після видалення стоп-слів і пунктуації:', len(cleaned))
    print('Топ-10 початкових слів:', raw_top)
    print('Топ-10 після очищення:', clean_top)
    print('Топ-10 після стемінгу:', stem_top)
    suffix = '_demo' if is_demo else ''
    chart(raw_top, f'text_top10_raw{suffix}.png', '10 найчастіших слів: вихідний текст')
    chart(clean_top, f'text_top10_clean{suffix}.png', '10 найчастіших слів: без стоп-слів')
    results = {
        'mode': 'demo excerpt' if is_demo else 'full gutenberg corpus',
        'word_count':len(tokens), 'unique_words':len(set(tokens)),
        'clean_count':len(cleaned), 'unique_clean':len(set(cleaned)),
        'top_original':raw_top, 'top_cleaned':clean_top, 'top_stemmed':stem_top
    }
    (OUT / f'blake_analysis{suffix}.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обробка творів Вільяма Блейка засобами NLTK')
    parser.add_argument('--demo', action='store_true', help='тестувати на справжньому фрагменті без завантаження')
    args = parser.parse_args()
    try:
        analyze(demo=args.demo)
    except (OSError, RuntimeError, ValueError) as e:
        print('ПОМИЛКА:', e)
        raise SystemExit(1)
