# Bot A — парный скальпинг и статистический арбитраж на MOEX (T-Invest API v2)

> **Управление простыми словами**: как запустить и остановить, где смотреть графики,
> что означают флаги — [MANUAL.md](./MANUAL.md).

Одно ядро стратегии и один конечный автомат исполнения для пяти режимов работы:

| Режим | Назначение |
| --- | --- |
| `preflight` | Проверка токена/счёта, готовности FORTS (`SPBFUT`), свободных средств и ГО, резолв `InstrumentUid`, шага цены и стоимости шага |
| `backtest` | Бэктест на минутных свечах T-Invest (с синтезом внутриминутных тиков) **или** на записанных L2-стаканах/сделках |
| `optimize` | Параллельный grid-search по `Z_Entry × Z_Exit × EwmaWindow × ExecutionTopology` с экспортом победителя |
| `record` | Запись live-потока стаканов (100 мс) и принтов сделок в PostgreSQL и/или `*.jsonl.gz` |
| `paper` / `live` | Торговля в реальном времени: внутренний симулятор исполнения или реальные заявки |

> **О языке реализации.** В технических требованиях указан Python (раздел «Tech Stack»), при этом часть
> пунктов описана в терминах .NET (`Microsoft.Extensions.Hosting`, `Parallel.ForEach`, `readonly record struct`).
> Проект реализован на **Python 3.10+** (рабочая папка — `Python scripts`, интерпретатор Python в окружении),
> а архитектурные паттерны C# воспроизведены один-в-один:
> `readonly record struct` → `@dataclass(frozen=True, slots=True)`,
> `System.Threading.Channels` → `bot_a/utils/channel.py`,
> DI/`IHostedService` → явная сборка зависимостей в `bot_a/engine/factory.py` + `asyncio` воркеры,
> Serilog (Console + File) → `logging` + `RotatingFileHandler`.

---

## 1. Установка

```powershell
# бэктест, реплей, отчёты, тесты
py -3 -m pip install -r requirements.txt

# живые режимы (preflight/record/paper/live) — нужен SDK T-Invest
py -3 -m pip install -r requirements-live.txt
```

**Важно:** для `tinkoff-invest-api` нужен Python **3.10–3.13** (у `grpcio`/`protobuf` нет колёс под 3.14).
Бэктест, оптимизация, реплей и тесты работают на любой поддерживаемой версии Python без SDK —
импорт SDK выполняется лениво, а живые режимы выдают понятную ошибку с подсказкой.

Токен задаётся одним из способов:

* `tInvest.token` в `appsettings.json`;
* переменная окружения `TINVEST_TOKEN` (имя настраивается в `tInvest.tokenEnvVar`);
* флаг `--token`.

```powershell
$env:TINVEST_TOKEN = "t.xxxxxxxx"
```

---

## 2. Быстрый старт

```powershell
# 1) Проверка счёта, прав, ГО и метаданных инструментов
py -3 -m bot_a --mode preflight --preset CNYRUBF_CRZ6

# 1b) Офлайн-проверка готовности к сессии (работает без токена): расписание и время
#     до открытия, согласованность параметров v2.2, тариф, ГО по всем пресетам, данные
py -3 -m bot_a --mode preflight --monday-ready

# 2) Бэктест на последних 7 днях минутных свечей (кэш в ./data_cache)
py -3 -m bot_a --mode backtest --preset CNYRUBF_CRZ6 --days 7 --tariff Premium

# бэктест без токена и без SDK — публичные свечи MOEX ISS (также --synth-mode close)
py -3 -m bot_a --mode backtest --source iss --preset SBERP_SBER --days 7 --tariff Premium

# запись L2 без SDK (Python 3.14): REST-клиент T-Invest, поллинг 150 мс, глубина 20
py -3 -m bot_a --mode record --preset VTBR_SBER --duration 3600 --record-interval-ms 150

# 3) Grid-search по сетке параметров (2 процесса по умолчанию, максимум 4)
py -3 -m bot_a --mode optimize --preset CNYRUBF_CRZ6 --days 14

# 4) Запись L2-потока на 2 часа (файлы + PostgreSQL, если задан DSN)
py -3 -m bot_a --mode record --preset SPZ6_SRZ6 --duration 7200

# 5) Бэктест по собственной записи
py -3 -m bot_a --mode backtest --source recordings `
  --recordings .\recordings\SPZ6_SRZ6_20250101_100000.jsonl.gz `
  --from 2025-01-01 --to 2025-01-01

# 6) Бумажная торговля на живом потоке, затем реальная
py -3 -m bot_a --mode paper --preset CNYRUBF_CRZ6 --duration 3600
py -3 -m bot_a --mode live  --preset CNYRUBF_CRZ6 --yes
```

Остановка — `Ctrl+C`: сигнал перехватывается, бот снимает все активные заявки и (по умолчанию)
закрывает позицию рынком, затем завершает работу.

---

## 3. Структура проекта

```
appsettings.json                  конфигурация + 4 готовых пресета пар
requirements.txt                  бэктест/реплей/тесты
requirements-live.txt             + SDK T-Invest
pyproject.toml                    метаданные пакета, настройки pytest

bot_a/
  cli.py                          CLI: --mode preflight|backtest|optimize|record|paper|live
  __main__.py                     python -m bot_a
  config.py                       типизированная конфигурация (appsettings.json)
  logging_setup.py                консоль + ротация файла (Serilog-аналог)
  models/domain.py                InstrumentMetadata, OrderBookSnapshot, SpreadMetrics,
                                  TradeRecord, DailyBacktestSummary, BacktestTotals, ...
  utils/
    quotation.py                  Quotation <-> Decimal, шаг цены, пункт -> рубли
    time_provider.py              ITimeProvider: реальные и симулированные часы (+ MSK)
    channel.py                    Channel (SingleReader) поверх asyncio.Queue
  economics/fees.py               прогрессивные тарифы Trader/Premium, IsEdgeSufficient
  math_engine/pair_spread.py      O(1) EWMA Z-score, weighted OBI (top-5), микропрайс
  execution/
    gateway.py                    IOrderExecutionGateway + события исполнения
    live_gateway.py               реальные заявки через OrdersService (поллинг состояния)
    simulated_gateway.py          латентность, мейкер/тейкер, очередь, слиппедж, комиссии
  risk/guard.py                   расписание MOEX, клиринг, ГО + 25%, дневной kill-switch
  engine/
    fsm.py                        PairScalpingStateMachine (DualLegHedged / SingleLegLeadLag)
    ledger.py                     дневная агрегация PnL, просадка, Sharpe, profit factor
    factory.py                    сборка ядра для всех режимов
  broker/tinvest.py               тонкий ленивый адаптер SDK (gRPC v2)
  backtest/
    candles.py                    минутные свечи + дневной кэш
    synthetics.py                 мостик O->L->H->C, 12 подтиков, top-5 стакан, принты
    historical_candles.py         режим 1: свечи API -> синтез -> ядро
    tick_replay.py               режим 2: реплей L2 из PostgreSQL / *.jsonl.gz
    recorder.py                   запись live-потока (PG + gzip JSONL)
    recording_format.py           формат записей и разбор строк PG
    reporting.py                  консоль, Markdown, CSV (дни и сделки), JSON
    grid_search.py                параллельный перебор, ранжирование, экспорт победителя
    simulation.py                 общий цикл прокрутки событий
  services/
    preflight.py                  проверки и отчёт preflight
    live_worker.py                paper/live воркер: стрим -> канал -> FSM -> заявки
tests/                           124 теста (pytest)
```

---

## 4. Конфигурация (`appsettings.json`)

| Секция | Содержимое |
| --- | --- |
| `tInvest` | токен, имя приложения, таймаут, `accountId` |
| `logging` | уровень, каталог и файл лога |
| `fees` | `tariff`: `Trader` / `Premium` |
| `risk` | дневной лимит убытка, буфер ГО (25 %), `maxPositionLots`, вечерняя сессия, флэт при остановке |
| `execution` | `topology`, латентность симулятора (25 мс), `queueFillFactor` (0.35), слиппедж (шаги), глубина стакана; v2.4.1: `statSampleIntervalSeconds` (5.0 — шаг статистических окон для live/paper/replay) и `bootstrapWarmupDays` (5 — прогрев из MOEX ISS перед сессией) |
| `strategy` | `zEntry` (2.8), `zExit` (0.2), `zCancel`, `zStop` (6.0), `ewmaWindow` (2400), `maxHoldSeconds` (1800), `minEdgeMultiplier`, `quantityLots`; v2: `useFrozenMu`, `useProfitLock`, `profitFloorMultiplier` (1.25), `dualProfitFloorMultiplier` (1.3), `ghostReversionZ` (0.3), `ghostFlatSeconds` (45), `scratchSeconds` (60), `betaMin/betaMax`, `useEntryZCap`, `entryZCapSlack`; v2.1: `betaMode` (`auto` → режим пресета), `minNetProfitMultiplier` (1.35), `tpDeviationFraction` (0.55), `timeStopQuoteSeconds` (30); v2.2: `useZPeakHook`, `zHookPullback` (0.12), `zMinTrigger` (0.15), `zPeakResetGap` (0.25), `useTpDecay`, `tpDecayStartFraction` (0.5), `tpDecayFloorMultiplier` (1.15), `useRegimeFilter`, `regimeWindow` (12), `regimeDominantShare` (0.65), `regimeStableShare` (0.35); v2.2.1: `useBreakevenRatchet`, `ratchetArmMultiplier` (2.2), `ratchetArmTargetFraction` (0.90), `ratchetArmStepBuffer` (1), `ratchetRetraceSteps` (2), `ratchetMinHoldSeconds` (120), `ratchetExitMultiplier` (1.15), `ratchetExitTargetFraction` (0.60); v2.3: `tpMinFeeMultiple` (2.5), `useTpDecay` (false), `tpDeviationFraction` (0.85), `useSingleLegPriceStop` (true), `singleLegStopToTargetRatio` (1.5), `singleLegStopMinSteps` (4), `leadFailureZ` (0.5), `leadFailureSeconds` (60); v2.4: `useDailyStopBreaker` (true), `maxDailyHardStops` (1), `maxDailyLossFeeMultiple` (3.5), `useMacroBetaDriftGuard` (true), `macroBetaWindow` (14400), `macroBetaMinSamples` (1800), `maxBetaDrift` (0.008), `useVolToFeeFilter` (true), `minSigmaToFeeRatio` (1.30) |
| `risk` | дневной лимит убытка, буфер ГО, `maxPositionLots`, вечерняя сессия, флэт при остановке, **`activeSessionWindows`**: `AllDay` / `MainSessionOnly` / `PrimeHoursOnly` |
| `storage` | PostgreSQL DSN, каталоги записей/кэша/отчётов, `stateDir` (`./state` — персистентный стейт дня), параметры батчей |
| `presets` | пары инструментов: `CNYRUBF/CRZ6`, `T_SBER` и `SBERP/SBER` (`enabled: false` — отключены от торговли), `SPZ6/SRZ6`, `IMOEXF/MMZ6`, `SBERP/SBER` (Z 3.2), `VTBR/SBER` (Z 2.8), `SVZ6/GDZ6`; у каждой пары свой `betaMode` (`same_underlying` для однородных, `slow_ratio` для кросс-активных) и опциональные переопределения `zEntry`, `topology`, `activeSessionWindows` |

Пресеты содержат шаблонные значения шага цены и ГО — **preflight перезаписывает их фактическими
данными из `InstrumentsService`** (`FutureBy`, `GetFuturesMargin`) перед торговлей.

Дефолты `strategy` соответствуют проверенному полугодовому прогону v2.4 (`Z_Entry` 2.8, `EWMA` 2400,
`MaxHold` 1800 с, `Z_Stop` 6.0, хук пика Z, выпуклый тейк без сжатия и ratchet, ценовой стоп Leg1,
дневной предохранитель, макро-дрейф беты, фильтр σ/Fee 1.30) — см.
`reports/semiannual_oos_validation_v2_4.md`. Для режима `--synth-mode close` окно задаётся в барах
(`--ewma-window 120…240`).

Пресет с `"enabled": false` (`CNYRUBF_CRZ6`, `T_SBER`, `SBERP_SBER`) остаётся в конфигурации для
исследований, но блокируется в `--mode paper|live`; список торговых пар отдаёт
`AppConfig.trading_preset_names()`.

Пресет может переопределять глобальные настройки: `"zEntry"`, `"topology"`, `"activeSessionWindows"`
(см. `AppConfig.with_preset_overrides`). Приоритет: глобальный конфиг → значения пресета → флаги CLI.
Именно так в одном `appsettings.json` сосуществуют `VTBR_SBER` (Z 2.8) и `SBERP_SBER` (Z 3.2).

> **Итог v2.3 (25.07–25.10.2026, трёхмесячный прогон):** дефолты переведены на payoff-convexity
> (`tpMinFeeMultiple 2.5`, `tpDeviationFraction 0.85`, `useTpDecay=false`, `useBreakevenRatchet=false`),
> ценовой стоп Leg1 для `SingleLegLeadLag` и внутридневные окна `PrimeHoursOnly`.
> `VTBR/SBER` SingleLeg Premium даёт **+3.24 RUB** за 60 торговых дней (PF 2.02, DD 0.93,
> 35/50 зелёных дней), Trader — +2.01 RUB. Подробности и абляция —
> [reports/quarterly_backtest_v2_3.md](reports/quarterly_backtest_v2_3.md).
> `T/SBER` остаётся убыточным (сигнал, а не исполнение), `CNYRUBF/CRZ6` — отключена.

> **Вердикт v2.3.1 (слепой OOS-квартал 25.04–25.07.2026):** OOS **не подтверждает** in-sample edge.
> `VTBR/SBER` Z 2.8: OOS **−1.79 RUB** (PF 0.80) против IS **+3.24 RUB** (PF 2.02); полугодие
> 25.04–25.10.2026 остаётся положительным только за счёт IS: **+1.46 RUB**, PF 1.12, Sharpe 0.76,
> DD 4.17 RUB, 63/101 зелёных дней (≈ **5.1 % годовых** при масштабировании на 50–500 тыс. RUB).
> `SBERP/SBER` Z 3.2 в OOS даёт **−6.71 RUB** (PF 0.15) и полугодие **−5.11 RUB** — в live не запускать.
> Отчёт и таблицы position sizing — [reports/semiannual_oos_validation_v2_3.md](reports/semiannual_oos_validation_v2_3.md).
> Дефолты зафиксированы (`VTBR_SBER` Z 2.8 / `SBERP_SBER` Z 3.2, Prime-окна), но до положительного
> OOS на 6+ месяцах и L2-реплея — только `--mode paper` и `--mode record`.

> **Итог v2.4 (те же 6 месяцев, 3 режимных предохранителя):** дневной предохранитель пары
> (`maxDailyHardStops=1` или убыток ≤ `−3.5·Fee`), макро-дрейф беты (`±0.8 %`, макро-окно 14400 тиков)
> и фильтр `σ_leg1/Fee ≥ 1.30`. `VTBR/SBER` Premium SingleLeg: **OOS впервые +0.22 RUB** (было −1.79),
> полугодие **+3.20 RUB** (PF 1.37, Sharpe 1.95, DD 2.17, 60/93 зелёных дней) против +1.46 у v2.3 —
> **≈ 11.2 % годовых** при просадке 3.8 % капитала. HardStop 13 → 8, покрытие gross/fee 1.30 → 1.79.
> `SBERP_SBER` **отключён** (`enabled: false`): комиссия 0.149 RUB/круг против 0.003 RUB gross.
> Подробности, абляция по механизмам и sizing — [reports/semiannual_oos_validation_v2_4.md](reports/semiannual_oos_validation_v2_4.md).
> Статус остаётся **paper/record-only**: OOS-плюс на 59 сделках статистически незначим.

> **v2.4.1 (боевая обвязка, математика не тронута):** статистические окна (`EWMA`, `β_slow`,
> `β_macro`, буфер `K=12`) двигаются по времени события раз в `statSampleIntervalSeconds = 5.0` —
> ровно как 12 подтиков/мин в бэктесте, тогда как микроструктура, `z_live`, котирование и стопы
> пересчитываются на каждом L2-пакете (100–200 мс). Перед `--mode paper|live` выполняется
> `bootstrap_warmup`: 5 торговых дней свечей MOEX ISS прогоняются через то же ядро с запретом
> торговли, поэтому бот входит в сессию в состоянии `Idle` с готовыми `μ/σ/β` (проверено на
> реальном кэше: `tick_count=50472`, `beta_drift=+0.003%`). Дневной предохранитель теперь
> персистентен: `./state/daily_state_{preset}.json`, подхват в тот же день MSK.
> 6-месячный бэктест воспроизводится **бит-в-бит** (122 сделки, net 3.20227750, PF 1.36647,
> DD 2.16917750). Подробности — [reports/production_v2_4_1_live_cadence.md](reports/production_v2_4_1_live_cadence.md).

Переопределения CLI: `--preset`, `--tariff`, `--topology`, `--z-entry`, `--z-exit`, `--ewma-window`,
`--quantity`, `--days`, `--from/--to`, `--source`, `--synth-mode`, `--recordings`, `--jobs`,
`--record-interval-ms`, `--record-depth`, `--log-level`, `--no-report`, `--no-flatten`, `--json`,
`--monday-ready`, `--charts`, `--chart-dir`, `--chart-trades`, `--port` (для `--mode watch`).

### Визуализация работы бота

`--charts` сохраняет в `reports/charts/<префикс>/` набор PNG и HTML-страницу:
обзор (цены обеих ног с точками входов ▲▼ и выходов ✕, Z-score с уровнями `Z_entry`/`Z_exit`/`Z_stop`
и подсветкой активных окон входа), кривую эквити с просадкой, дневной PnL, разбор сделок
(net по каждой сделке + MFE/фактический результат) и детальное приближение выбранных сделок
по 1-минутным свечам. В `index.html` — KPI-карточки и таблица всех сделок
(когда вошёл/вышел, цена, причина выхода, Z, MFE/MAE, комиссии, net); `trades.csv` — та же таблица.

### Живой стакан в браузере (`--mode watch`)

Режим показывает **реальный поток в реальном времени**: стакан обеих ног (глубина до 20),
микропрайс, OBI, спред, `μ/σ`, `β_slow`/`β_macro` и дрейф беты, Z-score со шкалой
`Z_entry`/`Z_exit`/`Z_stop`, состояние ФСМ, счётчики фильтров, таблицу сделок бота и ленту
принтов. Отдельная панель отвечает на вопрос «когда бот войдёт»: график Z с зонами входа и
зелёными маркерами моментов, когда все условия выполнены, плюс чек-лист из 8 фильтров
(`|Z| ≥ Z_entry`, хук пика Z, режимный фильтр, OBI, `σ/Fee`, покрытие комиссии, окно входа,
дневной предохранитель) с фактическими значениями и списком «что мешает прямо сейчас».
Исполнение — симулятор (деньги не тратятся), статистика прогревается из MOEX ISS перед стартом.

```powershell
py -3 -m bot_a --mode watch --preset VTBR_SBER --tariff Premium --port 8099 --record-interval-ms 300
# затем открыть http://127.0.0.1:8099/  (страница обновляется 2.5 раза в секунду)
```

Данные: `MarketDataService/GetOrderBook` + `GetLastTrades` через REST (работает на Python 3.14
без gRPC). JSON-снимок доступен по `GET /api/state` — удобно для внешних дашбордов.

Панель помнит весь торговый день, а не только последние минуты: `_DayTracker` копит
`zMin/zMax/maxAbs/maxAt/aboveThreshold/readySamples` (видно на странице строками
«за день макс |Z|», «за день выше порога», «за день готовность») и раз в минуту дописывает
точку в `state/watch_<пресет>_<дата>.jsonl`. Поэтому вопрос «почему за день не было входов»
разбирается по фактам: смотрите `day` в `/api/state` или файл трейла.

### Портфельный режим (`--preset ALL_ACTIVE`)

Все включённые в `appsettings.json` пары ведутся одним процессом: у каждой свой экземпляр
ядра, свой дневной предохранитель, свой `PairBookSyncer` и свой трекер дня.

```powershell
py -3 -m bot_a --mode watch --preset ALL_ACTIVE --tariff Premium --port 8099 --record-interval-ms 300
py -3 -m bot_a --mode paper --preset ALL_ACTIVE --tariff Premium --port 8099
```

Страница `http://127.0.0.1:8099/` показывает таблицу пар: состояние, `Z`, «до входа»,
`σ/Fee`, дрейф беты, OBI, сделки, net, максимум `|Z|` за день, отброшенные рассинхронные тики
(`skew`) и ошибки REST. Зелёная строка — все 8 фильтров входа выполнены прямо сейчас.
Раз в 30 секунд та же сводка уходит в лог.

### Единицы измерения: спред-рубли против рублей Leg1 (v2.6.1)

Спред считается как `S = V2 − β·V1`, где `V = price × lot_size`, поэтому `S` живёт
в масштабе **Leg2**. Всё, что относится к Leg1 (цель тейка, ожидаемый возврат, цена стопа),
обязано переводиться делением на бету:

```text
ΔP_Leg1 = |S − μ| / (β × lot_size × lots)
```

До v2.6.1 деление на β отсутствовало: тейк уезжал в β раз дальше нужного (на `VTBR/SBER`
β ≈ 5.1, на `MTLR/SBER` β ≈ 7.6), а фильтр покрытия комиссии сравнивал разные единицы.
Регресс закрыт тестами `tests/test_lot_size_scaling.py`.

### Персональные настройки пары (v2.6)
У каждой пары в `appsettings.json` можно задать своё окно статистики и свой порог
волатильность/комиссия — глобальные значения при этом не меняются:

```json
"MTLR_SBER": { "enabled": true, "zEntry": 2.8, "ewmaWindow": 900,
               "topology": "SingleLegLeadLag", "activeSessionWindows": "PrimeHoursOnly" },
"SPZ6_SRZ6": { "enabled": true, "zEntry": 2.8, "minSigmaToFeeRatio": 1.05,
               "topology": "SingleLegLeadLag", "activeSessionWindows": "PrimeHoursOnly" }
```

`ewmaWindow` нужен, когда у пары другая частота сделок (у `MTLR` комиссия `0.022 RUB` за круг,
поэтому короткое окно `900` выгоднее). `minSigmaToFeeRatio` нужен фьючерсным календарным
спредам: их `σ/Fee` структурно ниже `1.30`, и глобальный порог их душит.

`quantityLots` задаёт размер позиции пары (equal-notional портфель). Следите, чтобы
`risk.maxPositionLots` был **не меньше** самого большого `quantityLots`: риск-предохранитель
режет размер до потолка, и пара начнёт торговать меньше задуманного (в лог при этом
пишется предупреждение «Позиция урезана предохранителем»).

### Грид бэктестов (тихий режим вычислений)

```powershell
py -3 scripts/pair_grid.py --pairs VTBR/SBER ALRS/SBER `
    --ewma 900 1200 1600 2400 --z 2.6 2.8 --periods oos is 6m
```

Считает конфигурации последовательно и печатает таблицу «пара × период × окно × порог»:
сделки, gross/fees/net, PF, просадку. Прогон воспроизводит CLI бит-в-бит.

> ⚠️ **Почему по умолчанию один процесс.** Пул из 6–15 процессов поднимал все 16 потоков
> CPU на 100 %, и на этой машине отваливался видеодрайвер (приходилось перезагружаться).
> Теперь `--jobs` по умолчанию `1`, максимум `4`, BLAS-потоки внутри воркера ограничены
> одним, приоритет процесса понижен до `BELOW_NORMAL`, а между прогонами есть пауза
> `--pause-ms` (по умолчанию 150 мс). То же ограничение действует для
> `scripts/portfolio_sim.py` и `--mode optimize`: там `--jobs` по умолчанию `2` вместо
> `cpu_count - 1`. Явно просить больше четырёх процессов бессмысленно — значение обрежется.
> Однопроцессный прогон 6M по одной паре занимает единицы минут и грузит машину на ~10 %.

### Защита от фантомных выбросов (v2.5)

REST опрашивает ноги последовательно, поэтому пара «одна нога обновилась, вторая пришла из
таймаута» давала ложные скачки `Z` (26.10.2026 в 18:55:34 зафиксирован `Z = −2.298` при
спокойном рынке). Два независимых механизма:

* `PairBookSyncer` (`bot_a/services/l2_sync.py`) — отбрасывает тик, если нога не ответила,
  метки книг разошлись больше `1.5 × интервал опроса` или одна книга «замёрзла»; счётчик
  `skewSkips` виден в `/api/state`;
* подтверждение сигнала (`useEntryPersistence`) — вход только если `|Z| ≥ Z_entry` держится
  два подряд 5-секундных замера (минимум 3 с); одиночный выброс не армирует пик хука,
  счётчик `persistenceSkips`.

В бэктесте подтверждение выключено, поэтому исторические прогоны воспроизводимы.

### Разбор «почему не было сделок»

```powershell
py -3 scripts/z_diagnostics.py --date 2026-10-26          # один день
py -3 scripts/z_diagnostics.py --from 2026-10-20 --to 2026-10-26 --top 8
```

Скрипт проигрывает реальные свечи ISS через то же ядро и печатает почасовую раскладку `Z`
(минимум, максимум, сколько замеров выше порога), счётчики ФСМ, долю блокировок по каждому
из 8 фильтров и топ-моменты с полным чек-листом. Кэш свечей за **текущий** день обновляется
автоматически (`today_ttl_seconds = 120`), прошедшие дни неизменяемы.

```powershell
# Полное видение на реальных данных без токена: paper-путь + прогрев + графики
py -3 -m bot_a --mode paper --source iss --preset VTBR_SBER --tariff Premium `
    --from 2026-10-05 --to 2026-10-26 --report-stem paper_vtbr_oct
```

Режим `--mode paper --source iss` — это тот же paper-путь (то же ядро, тот же симулятор
исполнения, тот же тариф и те же окна входа), но вместо gRPC-потока проигрываются реальные
1-минутные свечи MOEX ISS, а перед сессией выполняется прогрев `bootstrap_warmup`.
Токен для него не нужен.

Без токена: `--mode paper --source iss` (реальные свечи ISS).
С токеном: `--mode paper` идёт на **живом стакане** через REST-поллинг
(`MarketDataService/GetOrderBook`, глубина 20, `--record-interval-ms` 100–250 мс),
если gRPC SDK недоступен (Python 3.14). Запуск одной командой — `scripts/run_session.ps1`.
Токен читается из `--token`, `$env:TINVEST_TOKEN`, `TINVEST_TOKEN_FILE` или
`secrets/tinvest_token.txt` (BOM и кавычки снимаются, в логи не печатается).

---

## 5. Экономика T-Invest

### 5.1. Прогрессивный тариф (`bot_a/economics/fees.py`)

| Тариф | Оборот за день, RUB | Ставка за сторону |
| --- | --- | --- |
| `Trader` | ≤ 5 000 000 | 0.04 % |
| `Trader` | 5 000 000 – 10 000 000 | 0.03 % |
| `Trader` | > 10 000 000 | 0.025 % |
| `Premium` | ≤ 5 000 000 | 0.025 % |
| `Premium` | 5 000 000 – 17 000 000 | 0.02 % |
| `Premium` | > 17 000 000 | 0.015 % |

Комиссия начисляется от **полной рублёвой стоимости контракта** на каждой стороне.
Накопленный дневной оборот переключает ступень «на лету»; при смене торгового дня (MSK) оборот
сбрасывается. Один и тот же объект `TinkoffFeeCalculator` используется и для оценки издержек
перед входом (`estimate_sequence`, без мутации состояния), и для фактического списания при филлах.

### 5.2. FORTS: пункты → рубли и ГО

```
ValueInRub(P)   = P / MinPriceIncrement * MinPriceIncrementAmount     (фьючерс)
ValueInRub(P)   = P * LotSize                                          (акция TQBR)
PnL_нога(RUB)   = (P_exit - P_entry) * rub_per_price_unit * qty * side
rub_per_price_unit = MinPriceIncrementAmount / MinPriceIncrement       (фьючерс)
```

`InitialMarginOnBuy/InitialMarginOnSell` берутся из `GetFuturesMargin`; перед входом проверяется
`free_cash >= ГО * 1.25` (буфер 25 %).

### 5.3. Топологии исполнения

* `DualLegHedged` — Leg 1 пассивно (maker), Leg 2 агрессивно (market/IOC) сразу после филла:
  полный круг = 4 стороны комиссии.
* `SingleLegLeadLag` — Leg 2 только как индикатор справедливой цены и момента, заявки только по Leg 1:
  полный круг = 2 стороны (комиссия вдвое ниже). Используется по умолчанию.

---

## 6. Математическое ядро (O(1) на тик)

* Взвешенный OBI по top-5: `w_i = exp(-0.5 * (i-1))`,
  `OBI = (V_bid - V_ask) / (V_bid + V_ask) ∈ [-1, +1]`.
* Микропрайс: `(V_bid * P_ask + V_ask * P_bid) / (V_bid + V_ask)`.
* Спред пары в рублях: `S_t = ValueInRub(Leg2, P_micro2) - beta * ValueInRub(Leg1, P_micro1)`.
* Бета (v2.1, `betaMode`) — **стационарная**, без тикового `cov(ΔV2,ΔV1)/var(ΔV1)` (он давал эффект Эппса
  и скачки спреда при изменении β на тысячную):
  * `same_underlying` — β фиксирована (одна и та же нога: `SBERP/SBER`, `CNYRUBF/CRZ6`, `IMOEXF/MMZ6`);
  * `slow_ratio` — β_t = V̄₂/V̄₁ по медленным EWMA рублёвых стоимостей (кросс-активные пары),
    **замораживается** на входе вместе с `mu`/`sigma` и размораживается только в `Cooldown`;
  * `static` / `dynamic` сохранены для воспроизведения v1/v2.
* EWMA-дисперсия и Z-score (после прогрева `N` тиков):

```
alpha = 2 / (N + 1)
mu_t  = alpha * S_t + (1 - alpha) * mu_{t-1}
V_t   = (1 - alpha) * (V_{t-1} + alpha * (S_t - mu_{t-1})^2)
Z_t   = (S_t - mu_t) / sqrt(max(V_t, 1e-12))
```

* Защита от комиссии:

```
E_gross = |S_t - mu_t| * (|Z_t| - Z_exit) / |Z_t|
C_total = Fee_round_trip + Slippage_buffer
сигнал валиден <=> E_gross >= MinEdgeMultiplier * C_total
```

* **v2.2 — хук пика Z (`UseZPeakHook`)**: вход разрешён не в момент пересечения `|Z| >= Z_entry`,
  а после **истощения импульса**: `|z_peak| >= Z_entry` и откат `|z_peak| − |Z_t| >= Z_HookPullback`
  (0.12) либо два тика подряд с `ΔZ` против знака `Z` при `|Z| >= Z_entry − Z_MinTrigger`.
  Пик сбрасывается, когда `|Z| < Z_entry − Z_PeakResetGap` (0.25).
* **v2.2 — режимный фильтр направления (только `SingleLegLeadLag`)**: по изменению рублёвых
  стоимостей ног за `K = 12` тиков (`ΔV₁, ΔV₂`) допускаются два режима:
  * **A (отстающая нога)**: нога 2 дала ≥ 65 % движения спреда, нога 1 не сдвинулась
    (`|ΔV₁| <= 0.35·|ΔV₂|` и `ΔV₂·Z > 0`) — входим в направлении ноги 2;
  * **B (изолированный выброс)**: нога 1 сдвинулась в одиночку при стабильной ноге 2
    (`|ΔV₂| <= 0.35·|ΔV₁|` и `ΔV₁·Z < 0`) — гасим выброс ноги 1;
  * иначе (обе ноги уходят в одну сторону сопоставимым темпом) вход блокируется.

Все структуры рынка — `frozen=True, slots=True` датаклассы (аналог `readonly record struct`),
никаких аллокаций списков на горячем пути.

---

## 7. Конечный автомат исполнения

| Состояние | Действие |
| --- | --- |
| `Warmup` | набор `N` тиков, Z-score не используется |
| `Idle` | проверка расписания, kill-switch, `IsEdgeSufficient`, ГО; вход при `Z >= Z_entry` и `OBI_Leg1 > -0.45` (лонг) или `Z <= -Z_entry` и `OBI_Leg1 < +0.45` (шорт); **v2.2:** плюс хук пика Z и режимный фильтр направления |
| `QuotingEntryLeg1` | лимит на `BestBid`/`BestAsk`; реквота при сдвиге ≥ `RequoteStepThreshold` шагов, не чаще 250 мс и только если `|Z| >= Z_entry - 0.15`; снятие при `|Z| < Z_cancel` или токсичном OBI (`|OBI| > 0.70` против котировки) |
| `HedgingEntryLeg2` | только `DualLegHedged`: отмена остатка Leg 1, marketable IOC / market по Leg 2 на исполненный объём; таймаут → повтор рынком |
| `InPosition` | `z_frozen = (S_t − mu_entry)/sigma_entry` (v2) — цель не «уезжает» за спредом; TP при `|z_frozen| <= Z_exit` **и** `unrealized_gross >= Fee_rt × 1.25` (DualLeg × 1.3) → `QuotingExitLeg1`; по `MaxHoldSeconds` (1800 с) — **мейкерский** тайм-стоп (v2.1). **v2.2:** MFE/MAE считаются на каждом тике; при `useTpDecay=true` цель тейка сжимается по времени (по умолчанию в v2.3 выключено). **v2.2.1:** `BreakevenRatchet` (только `SingleLegLeadLag`, по умолчанию **выключен** в v2.3) взводится при `MFE >= max(2.2·Fee_rt, 0.90·цель) + 1 шаг` и закрывает мейкером при откате ≥ 2 шагов до `<= max(1.15·Fee_rt, 0.60·цель)` после 120 с. **v2.3:** стоп зависит от топологии — `DualLegHedged` по `\|z_frozen\| >= Z_stop`; `SingleLegLeadLag` **только** по фактической просадке Leg1 (`target × 1.5`, не меньше 4 шагов цены) либо по развороту импульса Leg2 (`z_frozen` за `−0.5` против позиции при убытке Leg1 и удержании ≥ 60 с) |
| `QuotingExitLeg1` / `HedgingExitLeg2` | пассивный лимит на `BestAsk` (выход из лонга) / `BestBid` (из шорта) до `MaxExitQuoteSeconds` (12 с) либо `timeStopQuoteSeconds` (30 с) для тайм-стопа, затем рынок; закрытие Leg 2 → `Cooldown` (3 с) |
| `EmergencyFlatten` | снятие всех заявок и закрытие всех ног рынком, повтор при неисполнении |

Каждая сделка пишется в `PortfolioLedger` с валовым PnL, комиссиями, чистым PnL, ГО, **MFE/MAE**
(v2.2) и причиной выхода (`TakeProfit`, `Breakeven`, `TimeStop`, `Scratch`, `HardStop`).

---

## 8. Бэктест

### 8.1. Режим свечей (`--source api`)

1. `MarketDataService.GetCandles` постранично (окно 1 день) для обеих ног; ответы кэшируются
   в `./data_cache/<uid>_1min_YYYYMMDD.json` — повторные прогоны не обращаются к API.
2. Синтез (`--synth-mode ticks`, по умолчанию): для каждой свечи строится детерминированный путь
   `O → L → H → C` (или `O → H → L → C`), 12 подтиков по 5 секунд, top-5 стакан вокруг шага цены,
   OBI из тела свечи и знака подтика, принты с U-образным распределением объёма.
   **v2.1:** порядок экстремумов внутри минуты определяется **только ведущей ногой (Leg2)** и
   навязывается обеим ногам (`phase_lock = true`) — это устраняет фазовую инверсию, когда Leg1
   стоит на максимуме, а Leg2 на минимуме одного и того же бара.
3. `--synth-mode close` — zero-artifact режим: только реальные закрытия 1-минутных баров
   (стакан + принт по цене закрытия), `EwmaWindow` масштабируется в бары (120–240 = 2–4 часа).
4. Тот же `PairScalpingStateMachine` прогоняется через `SimulatedOrderExecutionGateway`.

### 8.2. Режим L2-реплея (`--source recordings`)

* Источники: `./recordings/*.jsonl.gz` (файл или каталог) либо таблицы PostgreSQL
  `order_book_ticks` / `trade_prints`.
* Латентность постановки/замены/снятия — 25 мс (настраивается).
* Мейкер: филл при `BestAsk <= P_limit` (для покупки) **или** при накоплении принтов по цене лимита
  в объёме `qty / QueueFillFactor` (0.35 → нужно ≈2.9× нашего объёма).
* Тейкер: VWAP по доступным уровням + `SlippageSteps` шагов; для IOC остаток снимается.

### 8.3. Отчёты

Консоль и файлы в `./reports/`:

* дневная таблица: `Дата | Сделок | Win % | Gross RUB | Fees RUB | Net RUB | Net % ГО | Max DD | Ср. удержание`;
* сводка: суммарный Net PnL, прибыльные дни (`5/5`), profit factor, годовой Sharpe (по дневному ряду),
  среднее удержание, fee drag, максимальная просадка;
* `*.md`, `*_daily.csv`, `*_trades.csv`, `*.json`.

Сравнительные отчёты по прогонам на данных MOEX ISS:
[reports/weekly_backtest_summary.md](reports/weekly_backtest_summary.md) (v1),
[reports/weekly_backtest_v2_summary.md](reports/weekly_backtest_v2_summary.md) (v2),
[reports/weekly_backtest_v2_1_summary.md](reports/weekly_backtest_v2_1_summary.md) (v2.1),
[reports/weekly_backtest_v2_2_summary.md](reports/weekly_backtest_v2_2_summary.md) (v2.2 + §10 v2.2.1),
[reports/biweekly_backtest_v2_2_1.md](reports/biweekly_backtest_v2_2_1.md) (v2.2.1 за две недели, 11–25.10.2026),
[reports/quarterly_backtest_v2_2_1.md](reports/quarterly_backtest_v2_2_1.md) (v2.2.1 за три месяца, 25.07–25.10.2026),
[reports/quarterly_backtest_v2_3.md](reports/quarterly_backtest_v2_3.md) (v2.3: выпуклый тейк, ценовой стоп Leg1, окна входа),
[reports/semiannual_oos_validation_v2_3.md](reports/semiannual_oos_validation_v2_3.md) (v2.3.1: слепой OOS-квартал, полугодие, position sizing).
[reports/semiannual_oos_validation_v2_4.md](reports/semiannual_oos_validation_v2_4.md) (v2.4: дневной стоп, макро-дрейф беты, фильтр σ/Fee на 6 месяцах),
[reports/production_v2_4_1_live_cadence.md](reports/production_v2_4_1_live_cadence.md) (v2.4.1: каденция 5 с, прогрев из ISS, персистентный стейт дня).
[reports/session_run_20261026.md](reports/session_run_20261026.md) (запуск на реальных данных: paper-сессии, графики входов/выходов, sizing).

---

## 9. Запись и формат данных

JSONL (по одному событию в строке, gzip):

```json
{"t":1741676400123,"k":"book","leg":1,"uid":"...","b":[["7.292","15"]],"a":[["7.293","20"]],"lp":"7.292"}
{"t":1741676400123,"k":"trade","leg":2,"uid":"...","p":"7.2925","q":4,"s":"buy"}
```

PostgreSQL: `order_book_ticks(ts_ms, instrument_uid, leg, bids jsonb, asks jsonb, last_price)` и
`trade_prints(ts_ms, instrument_uid, leg, price, quantity, side)` — схема создаётся автоматически
при старте записи. Запись идёт батчами (`recordingBatchSize`, `recordingFlushSeconds`).

Транспорт записи выбирается автоматически (`bot_a/cli.py::_run_record`):

1. `bot_a` gRPC SDK + токен → полноценные стримы `MarketDataStream` (100 мс);
2. токен без SDK (Python 3.14) → **REST-клиент** `bot_a/broker/tinkoff_rest_client.py`
   (`https://invest-public-api.tinkoff.ru/rest/`, `MarketDataService/GetOrderBook` глубиной
   `--record-depth`, поллинг `--record-interval-ms`, сделки с дедупликацией);
3. без токена → `IssRecordingService` (снимки стакана и сделки публичного MOEX ISS, интервал ≥ 1000 мс).

---

## 10. Оптимизация

```powershell
py -3 -m bot_a --mode optimize --preset IMOEXF_MMZ6 --days 14
```

Сетка по умолчанию: `Z_Entry ∈ {1.8, 2.0, 2.2, 2.5, 2.8}`, `Z_Exit ∈ {0.0, 0.2, 0.4}`,
`EwmaWindow ∈ {200, 400, 600}`, `ExecutionTopology ∈ {DualLegHedged, SingleLegLeadLag}` — 90 комбинаций.
Свечи загружаются один раз, комбинации считаются через `ProcessPoolExecutor`
(2 процесса по умолчанию, `--jobs` не поднимется выше 4 — см. раздел про тихий режим),
ранжирование: `(Net PnL, доля прибыльных дней, Sharpe)`. Победитель выгружается в
`reports/optimize_winner_<preset>_<ts>.json`, полная таблица — в `.md`.

---

## 11. Риск-контроль

* Расписание MOEX (MSK, UTC+3): FORTS `10:00–14:00`, `14:05–18:45`, `19:05–23:50`;
  TQBR `09:50–14:00`, `14:05–18:40`, `19:05–23:50`.
* Клиринг блокируется: `14:00–14:05` и `18:45–19:05`; выходные — блок.
* **Внутридневные окна входа (v2.3, `risk.activeSessionWindows`):** `AllDay` — вся сессия,
  `MainSessionOnly` — `10:03–18:35 MSK` (без вечерней сессии), `PrimeHoursOnly` —
  `10:05–12:30` и `15:30–18:35 MSK`. Фильтр ограничивает только новые входы; выходы и аварийное
  закрытие работают всегда (счётчик `sessionSkips` в статистике ФСМ).
* `--mode preflight --monday-ready` — офлайн-проверка перед сессией: время до ближайшего
  открытия, зазор `Z_stop − Z_entry`, включённость фильтров v2.2, калибровка ratchet,
  адекватность тарифа, ГО по всем пресетам с буфером 25 %, наличие токена/кэша ISS.
* Вечернюю сессию можно отключить (`risk.tradeEveningSession`).
* ГО: `free_cash >= InitialMargin * qty * 1.25`.
* Дневной kill-switch: при убытке `>= maxDailyDrawdownRub` новые входы запрещены до следующего дня.
* `maxPositionLots` ограничивает размер позиции независимо от `strategy.quantityLots`.

---

## 12. Тесты

```powershell
py -3 -m pytest -q
```

74 теста покрывают: преобразования `Quotation`/шаг цены/экономику фьючерсов, прогрессивные тарифы,
OBI/микропрайс/EWMA, стационарную бету (`slow_ratio` + заморозка), расписание и kill-switch,
латентность и логику филлов симулятора, полный круг сделки в FSM (single/dual leg, edge-guard,
profit-lock, ghost-reversion/скретч, мейкерский тайм-стоп), фильтры v2.2 (хук пика Z с порогами
0.12/0.25 и откатом по ΔZ, breakeven-ratchet по MFE ≥ 1.5·Fee, сжатие цели тейка по времени,
режимные кейсы A/B и блокировка тренда), проверку готовности к сессии (`monday-ready`),
фазово-синхронный мост и режим `close`, бэктест по свечам, REST-клиент T-Invest (на фейковом
транспорте), запись → реплей, grid-search и загрузку конфигурации.

---

## 13. Ограничения и предупреждения

* Значения ГО/шага в пресетах — шаблонные; **обязательно** выполняйте `preflight` перед торговлей.
* Праздничные дни MOEX не учитываются (только выходные и клиринговые окна).
* В live-режиме комиссия для учёта считается по модели тарифа; сверяйте с отчётом брокера.
* Живые режимы используют публичный API `tinkoff-invest-api` 1.x, если пакет установлен; на Python 3.14
  (нет wheel) `--mode record` автоматически переключается на REST-клиент `bot_a/broker/tinkoff_rest_client.py`
  (поллинг `GetOrderBook`, `GetCandles`, `PostOrder`/`CancelOrder`/`ReplaceOrder`), а `paper`/`live`
  требуют установленного SDK.
* Синтетический бэктест на свечах — приближение: реальное исполнение проверяйте на L2-реплее.
* Sandbox-счета T-Invest не поддерживаются.
* Проект не является инвестиционной рекомендацией. Торговля на срочном рынке с плечом может
  привести к полной потере средств.
