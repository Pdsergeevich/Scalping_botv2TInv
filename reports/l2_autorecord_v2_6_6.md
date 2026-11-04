# Патч v2.6.6 — автоматическая запись L2 в `--mode watch` / `--mode paper` (`ALL_ACTIVE`)

Дата: 27.10.2026. Реализует Приоритет №1 из `benchmark_close_aligned_v2_6_5.md`:
«записать 3–5 дней реального L2 и прогнать реплей, чтобы понять, где исполнение между
`ticks` и `close`».

**190 тестов зелёные**, `--mode preflight --monday-ready` — «К СЕССИИ ГОТОВ».

---

## 1. Что сделано

### 1.1. Суточный файл на пару

`MarketDataRecorder` получил режим `daily=True`: имя файла
`recordings\{ПАРА}_{YYYY-MM-DD}.jsonl.gz` (торговая дата по MSK), открытие в режиме
дописывания, автоматическая ротация при переходе через полночь.

### 1.2. Запись внутри портфельного раннера

`MultiPresetRunner(record_l2=True)`:

* каждая пара получает свой `MarketDataRecorder` (только файлы, без PostgreSQL);
* в `_poll_pair` снимки обеих ног пишутся **только если** `PairBookSyncer` подтвердил
  синхронность (`decision.ok`) — рассинхронные и пустые стаканы не сохраняются;
* при остановке сессии файлы закрываются, в лог уходит число записанных снимков и путь;
* счётчик `recordedBooks` и путь к файлу видны в `/api/state` и в дашборде
  (колонка «L2 записан», путь — в подсказке), в консольной сводке — колонка «запис.».

### 1.3. Управление

| флаг | поведение |
|---|---|
| по умолчанию | запись **включена** для `--preset ALL_ACTIVE` |
| `--no-record-l2` | выключить (экономия диска) |
| `--record-l2` | включить принудительно (одиночный пресет) |

Формат строки — общий с `--mode record` (`bot_a/backtest/recording_format.py`):

```json
{"t":1793137987800,"k":"book","leg":1,"uid":"...","b":[["57.5",40],...],"a":[["57.51",25],...],"lp":"57.5"}
```

Поэтому файл читается напрямую `TickReplayBacktester` (`--mode backtest --source recordings`).

---

## 2. Проверка

### 2.1. Юнит-тесты (`tests/test_l2_autorecord.py`, 5 тестов)

| тест | что проверяет |
|---|---|
| `test_minute_of_runner_output_is_replayable` | 12 циклов `_poll_pair` (минута стаканов) → файл дня → `JsonlRecordingSource` + `TickReplayBacktester` читает **все 24 события** (2 ноги × 12) без ошибок формата |
| `test_recording_disabled_writes_nothing` | `record_l2=False` не создаёт файлов и счётчик остаётся 0 |
| `test_skewed_books_are_not_recorded` | рассинхрон 2 с (больше лимита 1.5 интервала) → `skewSkips` растут, `recordedBooks = 0` |
| `test_daily_filename_matches_replay_convention` | имя файла `{ПАРА}_{YYYY-MM-DD}.jsonl.gz` для `VTBR_SBER` и `AFKS_SBER` |

### 2.2. Живая проверка (реальный REST, 90 секунд)

```
23:13:07  ALL_ACTIVE: пар 4 (AFKS_SBER, IRAO_SBER, MTLR_SBER, VTBR_SBER)
23:13:07  AFKS_SBER: запись L2 → recordings\AFKS_SBER_2026-10-27.jsonl.gz
23:13:12  IRAO_SBER: запись L2 → recordings\IRAO_SBER_2026-10-27.jsonl.gz
23:13:17  MTLR_SBER: запись L2 → recordings\MTLR_SBER_2026-10-27.jsonl.gz
23:13:21  VTBR_SBER: запись L2 → recordings\VTBR_SBER_2026-10-27.jsonl.gz
23:14:57  VTBR_SBER: запись L2 завершена — стаканов 112, файл ...\VTBR_SBER_2026-10-27.jsonl.gz
```

Все четыре пары: по 112 снимков, 0 рассинхронов, 0 ошибок REST, файлы 3.7–4.7 КБ
(вечерняя сессия, 56 циклов).

Реплей этой записи через CLI:

```powershell
py -3 -m bot_a --mode backtest --source recordings --preset VTBR_SBER `
    --recordings "recordings\VTBR_SBER_2026-10-27.jsonl.gz"
```

```
Реплей VTBR / SBER за начало .. 2026-10-27 23:15 MSK
Событий обработано: 112
Всего сделок: 0     Чистый PnL: 0.00 RUB
```

Ошибок формата нет, события совпадают с записанными один-в-один.

---

## 3. Как пользоваться (кампания на 3–5 дней)

```powershell
# каждый торговый день — одна и та же команда; файлы дописываются в суточные
py -3 -m bot_a --mode watch --preset ALL_ACTIVE --tariff Premium `
    --port 8099 --record-interval-ms 400 --duration 28800
```

По окончании периода — реплей каждой пары по её файлу:

```powershell
py -3 -m bot_a --mode backtest --source recordings --preset VTBR_SBER `
    --recordings "recordings\VTBR_SBER_2026-10-27.jsonl.gz" --charts
```

Затем сравнить три числа по одной и той же паре и периоду:
`ticks` (синтетический мост), `recordings` (реальные снимки REST) и `close`
(только закрытия баров). Первое даёт верхнюю границу, третье — нижнюю,
а запись покажет фактическое положение.

**Объём диска:** глубина 20, 2 ноги, ~2 цикла/с на пару ≈ **3–5 МБ в час на пару** после
gzip; 4 пары × 6 часов ≈ 100 МБ в день. Записи не попадают в git (`recordings/`
в `.gitignore`).

---

## 4. Файлы патча

| файл | изменение |
|---|---|
| `bot_a/backtest/recorder.py` | режим `daily`: имя `{пресет}_{YYYY-MM-DD}.jsonl.gz`, ротация по MSK-дате |
| `bot_a/services/multi_preset.py` | `record_l2`, запись синхронизированных стаканов, закрытие файлов, `recordedBooks` в payload/консоли/дашборде |
| `bot_a/cli.py` | флаги `--record-l2` / `--no-record-l2`, по умолчанию включено для `ALL_ACTIVE` |
| `tests/test_l2_autorecord.py` | **новый**, 5 тестов (включая полный цикл «минута → файл → реплей») |
| `README.md`, `MANUAL.md` | раздел про автозапись и реплей записи |
