"""Практична робота №8–9, варіант 5. Завдання 2: Монреаль, 2015 рік."""
from pathlib import Path
import argparse
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
OUT = BASE / 'output'
CHARTS = BASE / 'charts'
# Архівний CSV зі щоденними показами велосипедних лічильників Монреаля.
SOURCE_URL = ('https://donnees.montreal.ca/dataset/f170fecc-18db-44bc-b4fe-5b0b6d2c7297/'
              'resource/64c26fd3-0bdf-45f8-92c6-715a9c852a7b/download/comptagevelo20152.csv')
MIRROR_URL = ('https://raw.githubusercontent.com/5nguy/opendata/master/montreal/'
              'ckan/ville-de-montreal/infrastructures/velos-comptage/comptagevelo20152.csv')
CACHED = DATA / 'montreal_bikes_2015_monthly.csv'
RAW = DATA / 'montreal_bikes_2015_daily.csv'


def download_daily_data():
    error = None
    for url in (SOURCE_URL, MIRROR_URL):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            table = pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            if 'Date' not in table.columns or len(table) < 300:
                raise ValueError('Завантажено файл не з тією структурою або роком')
            RAW.write_bytes(response.content)
            return table
        except (requests.RequestException, ValueError, UnicodeError, pd.errors.ParserError) as exc:
            error = exc
    raise RuntimeError(f'Не вдалося отримати файл за HTTP: {error}')


def to_monthly(daily):
    dates = pd.to_datetime(daily['Date'].astype(str), dayfirst=True, errors='coerce')
    if dates.isna().all():
        raise ValueError('Не вдалося визначити дати у щоденних даних')
    table = daily.copy()
    table['month'] = dates.dt.month
    table = table.loc[dates.dt.year == 2015]
    # У початковому файлі є допоміжний порожній стовпець з часом.
    excluded = {'Date', 'month', 'Unnamed: 1', ''}
    columns = [c for c in table.columns if c not in excluded and not str(c).startswith('Unnamed:')]
    for c in columns:
        table[c] = pd.to_numeric(table[c], errors='coerce')
    return table.groupby('month', as_index=False)[columns].sum(min_count=1)


def analyze(fetch=False):
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    if fetch:
        daily = download_daily_data()
        monthly = to_monthly(daily)
        monthly.to_csv(CACHED, index=False, encoding='utf-8-sig')
        print('Одержано щоденний CSV із порталу Монреаля:', len(daily), 'рядків')
    elif RAW.exists():
        monthly = to_monthly(pd.read_csv(RAW, encoding='utf-8-sig'))
        print('Використано локальні щоденні дані:', RAW.name)
    else:
        monthly = pd.read_csv(CACHED, encoding='utf-8-sig')
        print('Використано збережені місячні підсумки 2015 року:', CACHED.name)
    if monthly['month'].nunique() != 12:
        raise ValueError('Потрібні дані за всі 12 місяців 2015 року')

    lanes = [x for x in monthly.columns if x != 'month']
    for lane in lanes:
        monthly[lane] = pd.to_numeric(monthly[lane], errors='coerce')
    print('\n=== ЗАВДАННЯ 2. ВЕЛОДОРІЖКИ МОНРЕАЛЯ, 2015 ===')
    print('Перші 5 рядків:\n', monthly.head().to_string(index=False))
    print('\nІнформація про DataFrame:')
    monthly.info()
    print('\nСтатистика:\n', monthly.describe().round(1).to_string())

    lane_totals = monthly[lanes].sum().sort_values(ascending=False)
    grand_total = int(lane_totals.sum())
    selected = list(lane_totals.head(3).index)
    print('\nКількість за всіма велодоріжками за 2015 рік:', f'{grand_total:,}'.replace(',', ' '))
    print('Три обрані велодоріжки:', ', '.join(selected))
    max_months = []
    for lane in selected:
        imax = monthly[lane].idxmax()
        mon, value = int(monthly.loc[imax, 'month']), int(monthly.loc[imax, lane])
        max_months.append({'lane':lane, 'month':mon, 'count':value})
        print(f'  {lane}: найбільше відвідувачів у місяці {mon} ({value:,})')

    monthly['all_lanes_total'] = monthly[lanes].sum(axis=1)
    monthly.to_csv(OUT / 'bikes_2015_monthly_with_total.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'lane':lane_totals.index, 'total_2015':lane_totals.values}).to_csv(
        OUT / 'bikes_2015_lane_totals.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(max_months).to_csv(OUT / 'bikes_2015_best_months.csv', index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(10.5, 5))
    for lane in selected:
        ax.plot(monthly['month'], monthly[lane], marker='o', linewidth=2, label=lane)
    ax.set_title('Монреаль: велосипедні потоки на трьох доріжках, 2015 рік')
    ax.set_xlabel('Місяць')
    ax.set_ylabel('Кількість проїздів за місяць')
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    chart = CHARTS / 'bike_2015_three_lanes.png'
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    print('Графік створено:', chart.relative_to(BASE))
    return grand_total, max_months


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Аналіз велосипедних лічильників Монреаля за 2015 рік')
    parser.add_argument('--fetch', action='store_true', help='завантажити оригінальний щоденний CSV')
    args = parser.parse_args()
    try:
        analyze(fetch=args.fetch)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        print('ПОМИЛКА:', exc)
        raise SystemExit(1)
