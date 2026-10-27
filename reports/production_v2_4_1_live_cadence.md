# v2.4.1 — боевая каденция 5 с, прогрев из ISS и персистентный дневной предохранитель

Дата: 26.10.2026. Патч закрывает три разрыва между бэктестером и live-потоком, найденные
в §9 отчёта `semiannual_oos_validation_v2_4.md`. Математика исторического бэктеста **не изменена**:
6-месячный прогон воспроизводится бит-в-бит (см. §5).

| Проблема | До v2.4.1 | После v2.4.1 |
| --- | --- | --- |
| Окна `EWMA 2400` / `macro 14400` / `K=12` в тиках | на L2 100–200 мс окно 2400 тиков = 4–8 минут → σ сжимается → `σ/Fee ≥ 1.30` блокирует все входы | окна двигаются по времени события раз в 5.0 с: EWMA = 200 мин, макро-бета = 20 ч, режим = 60 с — как в бэктесте |
| Холодный старт в 10:00 MSK | 200 минут прогрева → весь `PrimeHoursOnly` (10:05–12:30) пропущен | прогрев 5 торговых дней из MOEX ISS, к 10:00 ядро в `Idle` |
| Дневной предохранитель | сбрасывался при рестарте процесса | `./state/daily_state_{preset}.json`, подхват в тот же день MSK |

## 1. Time-gated статистика при full-speed исполнении

`PairSpreadCalculator` получил параметр `stat_sample_interval_seconds`:

| Что происходит на **каждом** L2-пакете (100–200 мс) | Что происходит раз в **5.0 с** |
| --- | --- |
| микроструктура стакана (top-5 OBI), микропрайс | `EWMA(μ, V)` по спреду `S_t` |
| `S_t = V₂ − β·V₁` на текущей β | `β_slow` (`slow_ratio`, окно 2400) |
| мгновенный `z_live = (S_t − μ_t)/σ_t` | `β_macro` (окно 14400) |
| котирование, `z_frozen`, все стопы, `TimeStop`, ratchet | буфер режимного фильтра `K = 12` |

Ключевое: `z_live` считается **на каждом тике** по текущему спреду и «старым» `μ/σ`, поэтому
исполнение и риск-контроль не теряют реактивность; двигаются только статистические оценки.
Флаг сэмпла попадает в `SpreadMetrics.stat_sampled`, и `_update_regime` пополняет буфер
`K = 12` только на сэмплах — иначе на L2 «12 тиков» означало бы 1.2–2.4 секунды вместо 60.

| Файл | Изменение |
| --- | --- |
| `bot_a/math_engine/pair_spread.py` | `stat_sample_interval_seconds`, `_stat_due()`, мгновенный Z вне сэмплов, `beta_macro`/`beta_drift`/`stat_sampled` в метриках |
| `bot_a/models/domain.py` | `SpreadMetrics.stat_sampled: bool = True` |
| `bot_a/engine/fsm.py` | `StrategyParams.stat_sample_interval_seconds` (по умолчанию **0.0** — путь бэктестера) |
| `bot_a/backtest/tick_replay.py` | реплей L2 берёт каденцию из `execution.statSampleIntervalSeconds` |
| `appsettings.json` | `execution.statSampleIntervalSeconds = 5.0` |

**Почему бэктест не меняется:** `HistoricalCandleBacktester` продолжает работать с
`stat_sample_interval_seconds = 0.0` — окно двигается на каждом снимке стакана, как в v2.4.
Квантование включается только в live/paper/replay. При интервале 0 новые ветки кода не исполняются.

## 2. Автоматический прогрев из MOEX ISS

`bot_a/services/bootstrap.py`:

* берёт последние `execution.bootstrapWarmupDays` (по умолчанию **5**) торговых дней
  через существующий `IssCandleSource` + `data_cache`;
* гоняет их через `iter_candle_events` (12 подтиков/мин = 5.0 с) в **тот же** экземпляр ядра;
* торговля на это время запрещена флагом `bootstrap`: `_maybe_enter` только наполняет
  статистику, но не открывает сделок (`engine.set_bootstrap(True/False)`);
* пишет в лог строку требуемого формата:

```text
Warmup complete: tick_count=50472, mu=0.4116, sigma_leg1=0.1076, beta_slow=5.122812,
beta_macro=5.122642, beta_drift=+0.003% (state=Idle, свечей 4203/4206, 2026-10-20T00:00..2026-10-24T02:49)
```

Это **реальный** вывод `bootstrap_warmup` на кэше MOEX ISS для `VTBR/SBER` (5 торговых дней
20–24.10.2026): ядро прогрето, `state=Idle`, `beta_drift` +0.003 % — далеко от порога 0.8 %,
сделок во время прогрева 0. Прогрев занимает единицы секунд и идёт до подписки на стрим,
поэтому к 10:00:00 MSK бот готов, а первые сигналы в 10:05 MSK уже валидны.

Вызывается из `LiveTradingWorker.run()` до создания стрима (`self._bootstrap(...)`),
результат попадает в итоговый snapshot (`warmup`). Любая ошибка прогрева не блокирует торговлю —
логируется предупреждение, и бот стартует как раньше (с обычным `Warmup`).

## 3. Персистентный дневной предохранитель

* Путь: `storage.stateDir` (по умолчанию `./state`) → `daily_state_{preset}.json`.
* Формат: `{ "date": "YYYY-MM-DD", "daily_hard_stops": int, "daily_net_rub": float, "daily_locked_reason": str }`
  (плюс имя пары в имени файла).
* Запись — при каждом закрытии сделки (`_register_daily_outcome`) и при срабатывании блокировки.
* Чтение — в конструкторе ядра и при смене торговой даты; применяется **только** если
  `date == текущая дата MSK`.
* В `backtest`/`optimize` путь не передаётся → поведение чисто in-memory, файлов не появляется.

## 4. Тесты (`tests/test_v241.py`, 12 тестов)

| Тест | Что проверяет |
| --- | --- |
| `test_stat_sampling_quantises_ewma_to_five_seconds` | 12 с по 100 мс → окно сдвинулось ровно 3 раза (t=0, 5, 10 с) |
| `test_stat_sampling_keeps_instant_z_on_every_tick` | между сэмплами `μ/σ` не меняются, но `z_live` пересчитывается и растёт |
| `test_quantised_sampling_does_not_change_per_call_legacy_path` | интервал 0 = 59 обновлений (путь бэктеста), 5 с = 30 |
| `test_regime_buffer_fills_only_on_stat_samples` | 20 с по 100 мс → +4 сэмпла в буфере `K`, а не +200 |
| `test_stops_still_evaluated_on_every_l2_tick` | `HardStop` по Leg1 срабатывает на 200-мс тиках между сэмплами |
| `test_bootstrap_warmup_makes_engine_ready_and_idle` | прогрев → `is_warm=True`, `state=Idle`, `beta_macro>0`, 0 сделок |
| `test_bootstrap_warmup_blocks_trading_while_running` | во время прогрева сильный сигнал не открывает позицию, после — открывает |
| `test_daily_state_persists_across_restart` | три «процесса»: HardStop → файл → рестарт в тот же день (блокировка жива) → следующий день (снята) |
| `test_daily_state_not_persisted_without_path` | в бэктесте файлов `daily_state_*.json` не создаётся |
| `test_live_worker_daily_state_path_and_cadence` | `./state/daily_state_VTBR_SBER.json` + каденция из конфига |
| `test_readiness_reports_cadence_and_warmup` | чек `Каденция и прогрев (v2.4.1)` в `--monday-ready` |
| `test_stat_sampling_config_defaults` | `5.0` с / `5` дней / `./state` в конфиге, `0.0` в стратегии |

## 5. Верификация: 6-месячный бэктест бит-в-бит

```text
py -3 -m bot_a --mode backtest --source iss --preset VTBR_SBER --tariff Premium \
    --from 2026-04-25 --to 2026-10-25 --json --no-report
```

| Метрика | v2.4 (эталон) | v2.4.1 | Совпадение |
| --- | --- | --- | --- |
| Сделок | 122 | **122** | ✅ |
| Gross, RUB | 7.240 | **7.240** | ✅ |
| Fees, RUB | 4.03772250 | **4.03772250** | ✅ |
| Net, RUB | 3.20227750 | **3.20227750** | ✅ |
| Max DD, RUB | 2.16917750 | **2.16917750** | ✅ |
| Profit Factor | 1.36647 | **1.3664740231687185** | ✅ |
| Sharpe | 1.94864 | **1.9486405516337555** | ✅ |
| Дней в плюсе | 60/93 | **60/93** | ✅ |
| Счётчики | 3321/737/320/1566/8096/1642/1287/719/791 | **идентичны** | ✅ |

Дополнительно:

* `py -3 -m pytest` → **107 passed** (95 → 107, +12 тестов v2.4.1).
* `py -3 -m bot_a --mode preflight --monday-ready` → «К СЕССИИ ГОТОВ» с новым чеком:
  `Каденция и прогрев (v2.4.1): сэмпл раз в 5.0 с: EWMA 200 мин, макро-бета 20.0 ч, режим 60 с;
  прогрев 5 торг. дней из MOEX ISS; стейт дня ./state`.
* Smoke-прогон `bootstrap_warmup` на реальных свечах ISS: 4203/4206 свечей, `tick_count=50472`,
  `is_warm=True`, `state=Idle`, 0 сделок.

## 6. Что ещё нужно сделать перед live

1. `β_macro` теперь измеряется в секундах, а не в «пакетах» — но окно 14400 сэмплов всё ещё
   равно 20 часам; при переходе на live стоит явно зафиксировать его как «3 торговые сессии»
   и проверить, что прогрев из 5 дней даёт стабильный дрейф.
2. Прогрев использует 1-минутные свечи ISS: внутриминутная микроструктура синтетическая.
   σ после прогрева (0.108 RUB на VTBR) — оценочная; на реальном L2 её нужно сверить через
   `--mode record` (фильтр `σ/Fee ≥ 1.30` чувствителен к уровню σ).
3. Проверка на данных — только paper: `--mode paper --preset VTBR_SBER --duration 3600` в
   `PrimeHoursOnly`, затем сравнить счётчики `sessionSkips/volFilterSkips` с бэктестом.
4. Состояние дня не переживает смену `stateDir` — при нескольких инстансах держите общий каталог.
