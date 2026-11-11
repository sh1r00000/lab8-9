"""Практична робота №8–9, варіант 5. Завдання 1: pandas та World Bank."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'data' / 'gdp_per_capita_ukr_usa_1991_2019.csv'
OUTPUT = BASE / 'output'


def analyze(source=SOURCE, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(source, encoding='utf-8-sig')
    except (OSError, UnicodeError, pd.errors.ParserError) as e:
        raise RuntimeError(f'Не вдалося прочитати файл {source}: {e}') from e

    required = {'country', 'country_code', 'year', 'gdp_per_capita_current_usd'}
    if not required.issubset(df.columns):
        raise ValueError(f'У CSV бракує стовпців: {sorted(required - set(df.columns))}')
    if df.empty:
        raise ValueError('Файл CSV не містить записів')

    df['year'] = pd.to_numeric(df['year'], errors='raise').astype(int)
    df['gdp_per_capita_current_usd'] = pd.to_numeric(df['gdp_per_capita_current_usd'], errors='coerce')
    df = df.dropna(subset=['gdp_per_capita_current_usd']).sort_values(['country_code','year']).reset_index(drop=True)
    if df.empty:
        raise ValueError('Немає коректних числових значень')

    print('=== ЗАВДАННЯ 1. БАЗОВИЙ АНАЛІЗ CSV ===')
    print('Повний DataFrame:')
    print(df.to_string(index=False))
    print('\nПерші 3 рядки:\n', df.head(3).to_string(index=False))
    print('\nТипи даних:\n', df.dtypes.to_string())
    print('\nРозмір DataFrame (рядки, стовпці):', df.shape)
    print('\nОписова статистика:\n', df.select_dtypes(include='number').describe().round(2).to_string())

    # Розрахункова ознака — зміна показника відносно попереднього року для країни.
    df['growth_percent'] = df.groupby('country_code')['gdp_per_capita_current_usd'].pct_change(fill_method=None) * 100
    filtered = df.loc[df['year'] >= 2000].copy()
    sorted_df = df.sort_values('gdp_per_capita_current_usd', ascending=False).copy()

    by_country = df.groupby(['country_code', 'country'], as_index=False).agg(
        years=('year', 'nunique'),
        mean_gdp=('gdp_per_capita_current_usd', 'mean'),
        min_gdp=('gdp_per_capita_current_usd', 'min'),
        max_gdp=('gdp_per_capita_current_usd', 'max'),
    )
    by_year = df.groupby('year', as_index=False).agg(
        countries=('country_code','nunique'),
        mean_gdp=('gdp_per_capita_current_usd','mean'),
        max_gdp=('gdp_per_capita_current_usd','max'),
    )
    print('\nФільтр за роком >= 2000: рядків', len(filtered))
    print('Перші 3 відсортовані рядки:\n', sorted_df.head(3).to_string(index=False))
    print('\nГрупування за країною:\n', by_country.round(2).to_string(index=False))
    print('\nУнікальних країн:', df['country_code'].nunique())

    df.to_csv(output / 'gdp_processed.csv', index=False, encoding='utf-8-sig')
    filtered.to_csv(output / 'gdp_filtered_from_2000.csv', index=False, encoding='utf-8-sig')
    sorted_df.to_csv(output / 'gdp_sorted_desc.csv', index=False, encoding='utf-8-sig')
    by_country.to_csv(output / 'gdp_grouped_by_country.csv', index=False, encoding='utf-8-sig')
    by_year.to_csv(output / 'gdp_grouped_by_year.csv', index=False, encoding='utf-8-sig')
    print('\nРезультати збережено в папці output/')
    return df, by_country


if __name__ == '__main__':
    try:
        analyze()
    except (ValueError, RuntimeError) as err:
        print('ПОМИЛКА:', err)
        raise SystemExit(1)
