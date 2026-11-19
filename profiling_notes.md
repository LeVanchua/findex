# Профілювання findex перед оптимізацією

## Докази та порядок вимірювання

1. `docs/profile_index_before.prof` — `cProfile` послідовного `findex index data` на 1000 документах.
2. `docs/profile_search_before.prof` — `cProfile` CLI-пошуку за `python OR база OR тестування`.
3. `docs/profile_search_workload.prof` — 1500 послідовних викликів оригінального `ranked_search`, щоб виключити разову ініціалізацію процесу.
4. `docs/flamegraph_search_cprofile.svg` — flame graph із цих статистик, збудований `flameprof`.
5. `docs/scalene_search.json` — профіль `Scalene 2.4.0` для старого Python-скорера; Python-vs-native розділення часу.

## Три дорогі ділянки

| Місце | Підтвердження | Тип витрат | Гіпотеза |
|---|---|---|---|
| Токенізація під час `index` | 359000 викликів `tokenize`, 0.249 с cumulative | Python об'єкти та регулярні вирази | Зменшити кількість тимчасових об’єктів або векторизувати частину обробки |
| Читання індексу перед пошуком | `open_index`: 0.107 с у CLI-профілі | Python десеріалізація + I/O | Тримати індекс у пам'яті через lifespan; для великих корпусів використовувати memmap |
| Власне ранжування | 1200 проходів generator expression, 0.006 с у CLI-профілі | Python цикл по постінгах | Зберігати doc_id/tf як масиви та обчислювати через NumPy |

Профілі CLI включають імпорт бібліотек і початкове завантаження, тому не слід підміняти ними `bench_lab08.py` для швидкості скорера. Мікробенчмарк тримає вже завантажений індекс і вектори в пам'яті.

## Обмеження py-spy

Спроба:

```bash
uvx --from py-spy py-spy record --format flamegraph --output docs/flamegraph_search.svg -- .venv/bin/python profile_workload.py search
```

Завершилася системною помилкою `EPERM: Operation not permitted` під час спроби призупинити дочірній процес. Через це не можна називати наявний SVG результатом `py-spy`: його створено з cProfile через `flameprof`.

На Windows після встановлення py-spy запустіть (за потреби PowerShell від адміністратора):

```powershell
uv tool run --from py-spy py-spy record --format flamegraph --output docs/flamegraph_pyspy.svg -- python profile_workload.py search
```

Після цього можна додати справжній `py-spy` SVG окремим комітом.
