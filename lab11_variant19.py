"""Лабораторная работа № 11. Вариант 19. Ситникова Ольга Алексеевна."""
import platform  # сведения о компьютере
import random  # генерация текста
import re  # удаление знаков препинания при нормализации
import time  # замер времени
import matplotlib.pyplot as plt  # графики

PI_PATTERN, SIZES, THRESHOLD = "abcbabc", [10**4, 5 * 10**4], 0.85  # образец для ручного расчёта π, длины текста и порог сходства (вариант 19)
WORDS = ("и в на что с по для не как из к от это при за о а но или все данных система информация алгоритм процесс анализ задача результат "
         "метод проект работа время код программа модель структура значение связь сеть запись поиск текст строка память таблица граф ключ "
         "ответ расчёт срок обработка управление решение качество требование ресурс компания клиент рынок договор платёж отчёт документ "
         "учёт склад заказ поставка сотрудник команда план цель риск оценка контроль доступ сервер сообщение файл версия ошибка проверка "
         "тест функция список очередь стек дерево путь вершина ребро вес цикл порядок сумма разность").split()  # словарь: частые слова идут первыми
WEIGHTS = [1 / (i + 5) for i in range(len(WORDS))]  # закон Ципфа со сдвигом ранга: вес слова убывает с рангом
LEGAL_FORMS = {"ооо", "оао", "зао", "пао", "ао", "ип"}  # организационно-правовые формы для удаления


def natural_text(n, seed=7):  # модель естественного текста: предложения из слов с частотами по закону Ципфа
    rng, parts, length = random.Random(seed), [], 0  # генератор, части текста и текущая длина
    while length < n:  # пока текст короче n
        sentence = " ".join(rng.choices(WORDS, WEIGHTS, k=rng.randint(6, 14))).capitalize() + ". "  # предложение из 6…14 слов
        parts.append(sentence)  # добавляем предложение
        length += len(sentence)  # увеличиваем длину
    return "".join(parts)[:n]  # обрезаем текст до n символов


def naive_search(text, pattern):  # наивный поиск: позиции и число сравнений символов
    n, m, positions, comparisons = len(text), len(pattern), [], 0  # длины, найденные позиции и счётчик
    for i in range(n - m + 1):  # образец прикладывается к каждой позиции
        j = 0  # число совпавших символов
        while j < m:  # посимвольное сравнение
            comparisons += 1  # считаем сравнение
            if text[i + j] != pattern[j]:  # несовпадение
                break  # переходим к следующей позиции
            j += 1  # символ совпал
        if j == m:  # совпал весь образец
            positions.append(i)  # запоминаем позицию
    return positions, comparisons  # результат


def prefix_function(pattern):  # префикс-функция: pi[i] — длина наибольшего собственного префикса P[0..i], равного его суффиксу
    pi, k = [0] * len(pattern), 0  # значения и длина текущего совпадения
    for i in range(1, len(pattern)):  # перебираем символы образца
        while k > 0 and pattern[i] != pattern[k]:  # несовпадение
            k = pi[k - 1]  # откатываемся к более короткому префиксу
        if pattern[i] == pattern[k]:  # символы совпали
            k += 1  # увеличиваем длину совпадения
        pi[i] = k  # записываем значение
    return pi  # префикс-функция


def kmp_search(text, pattern):  # алгоритм Кнута — Морриса — Пратта, Θ(n + m)
    pi, positions, comparisons, k = prefix_function(pattern), [], 0, 0  # префикс-функция, позиции, счётчик и длина совпадения
    for i, ch in enumerate(text):  # указатель по тексту не возвращается назад
        while k > 0 and ch != pattern[k]:  # несовпадение после k совпавших символов
            comparisons += 1  # считаем сравнение
            k = pi[k - 1]  # сдвигаем образец по префикс-функции
        comparisons += 1  # сравнение символа текста с pattern[k]
        if ch == pattern[k]:  # символы совпали
            k += 1  # продолжаем совпадение
        if k == len(pattern):  # найдено вхождение
            positions.append(i - len(pattern) + 1)  # запоминаем позицию начала
            k = pi[k - 1]  # готовимся к следующему вхождению
    return positions, comparisons  # результат


def edit_table(a, b, w_ins=1, w_del=1, w_rep=1):  # полная таблица редакционного расстояния a -> b с весами операций
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]  # таблица (len(a)+1) × (len(b)+1)
    for i in range(1, len(a) + 1):  # первый столбец: удаление i символов
        D[i][0] = i * w_del  # стоимость удаления
    for j in range(1, len(b) + 1):  # первая строка: вставка j символов
        D[0][j] = j * w_ins  # стоимость вставки
    for i in range(1, len(a) + 1):  # заполняем таблицу по строкам
        for j in range(1, len(b) + 1):  # по столбцам
            D[i][j] = min(D[i - 1][j] + w_del, D[i][j - 1] + w_ins, D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else w_rep))  # удаление, вставка, замена
    return D  # таблица


def levenshtein(a, b, w_ins=1, w_del=1, w_rep=1):  # расстояние Левенштейна, Θ(nm) времени и Θ(len(b)) памяти
    prev = [j * w_ins for j in range(len(b) + 1)]  # предыдущая строка таблицы
    for i, ca in enumerate(a, start=1):  # по символам a
        cur = [i * w_del]  # текущая строка начинается с удаления i символов
        for j, cb in enumerate(b, start=1):  # по символам b
            cur.append(min(prev[j] + w_del, cur[j - 1] + w_ins, prev[j - 1] + (0 if ca == cb else w_rep)))  # удаление, вставка, замена
        prev = cur  # переходим к следующей строке
    return prev[-1]  # расстояние


def similarity(a, b, **w):  # нормированная мера сходства из [0; 1]
    if not a and not b:  # обе строки пусты
        return 1.0  # полное сходство
    return max(0.0, 1 - levenshtein(a, b, **w) / max(len(a), len(b)))  # 1 - d / max(длин), не меньше 0


def normalize(name):  # нормализация наименования контрагента
    s = re.sub(r"[\"'«»„“”.,()\-]", " ", name.lower().replace("ё", "е"))  # нижний регистр, «ё» -> «е», удаление знаков
    return " ".join(w for w in s.split() if w not in LEGAL_FORMS)  # удаление организационно-правовых форм и лишних пробелов


def match_records(list_a, list_b, threshold, **w):  # сопоставление двух списков наименований
    norm_b, result = [(b, normalize(b)) for b in list_b], []  # нормализованный список B и результат
    for a in list_a:  # для каждой записи системы А
        na, best, best_sim, kind = normalize(a), None, 0.0, ""  # нормализованное имя и лучший кандидат
        for b, nb in norm_b:  # перебираем записи системы Б
            if na == nb:  # точное совпадение после нормализации
                best, best_sim, kind = b, 1.0, "точное"  # фиксируем и выходим
                break  # лучшего не бывает
            s = similarity(na, nb, **w)  # нечёткое сходство
            if s > best_sim:  # кандидат лучше текущего
                best, best_sim, kind = b, s, "нечёткое"  # запоминаем
        result.append((a, best, best_sim, kind if best_sim >= threshold else "не найдено"))  # решение по порогу
    return result  # список (запись А, кандидат Б, сходство, тип)


def best_time(func, repeats):  # лучшее время func из нескольких повторов, с
    times = []  # времена повторов
    for _ in range(repeats):  # повторы
        t0 = time.perf_counter()  # начало замера
        func()  # запуск
        times.append(time.perf_counter() - t0)  # длительность
    return min(times)  # наименее искажённое помехами значение


SYSTEM_A = ['ООО «Ромашка»', "АО Василёк", 'ООО «Северный ветер»', 'ПАО «Технопром»', 'ООО «Гелиос-Плюс»', 'АО «Прогресс»', 'ООО «Меридиан»',
            "ИП Петров П.П.", 'ООО «Альфа»', 'ЗАО «Восход»', 'ООО «Строительная компания Юг»', 'ПАО «Гранит»']  # записи системы А
SYSTEM_B = ["Ромашка, ООО", "ВАСИЛЕК АО", "Северный Ветер ООО", "Технопорм ПАО", "Гелиос Плюс ООО", "ПАО Прогрес", "Мередиан ООО",
            "Петров Пётр Петрович", "Альфа-Бета ООО", "Восхот ЗАО", "Строительная компания Юк ООО", "Гранит-Строй ПАО"]  # записи системы Б

if __name__ == "__main__":  # запуск только при прямом старте файла
    print(platform.platform(), "|", platform.processor() or platform.machine(), "| Python", platform.python_version())  # система
    print(f"\nПрефикс-функция для '{PI_PATTERN}':", prefix_function(PI_PATTERN))  # результат для сверки с ручным расчётом
    assert prefix_function(PI_PATTERN) == [0, 0, 0, 0, 1, 2, 3]  # совпадает с ручным расчётом из отчёта
    a, b = "кошка", "окошко"  # пара коротких слов для таблицы
    print(f"\nТаблица редакционного расстояния '{a}' -> '{b}':")  # заголовок таблицы
    print("      ∅  " + "  ".join(b))  # заголовок столбцов
    for i, row in enumerate(edit_table(a, b)):  # строки таблицы
        print(f"  {'∅' if i == 0 else a[i - 1]}  " + "  ".join(f"{x}" for x in row))  # строка с подписью
    print(f"  d = {levenshtein(a, b)}, с весом замены 2: d = {levenshtein(a, b, w_rep=2)}")  # расстояния
    pattern, repeats = "алгоритм", ("a" * 50 + "b")  # образцы для естественного текста и текста с повторами
    print("\n      n |        текст | наивный, с |  КМП, с | сравнений (наив.) | сравнений (КМП) | вхождений")  # заголовок таблицы
    rows = []  # результаты для графика
    for n in SIZES:  # длины текста
        for label, text, p in (("естественный", natural_text(n), pattern), ("с повторами", "a" * n, repeats)):  # два типа текста
            (pos1, c1), (pos2, c2) = naive_search(text, p), kmp_search(text, p)  # оба алгоритма
            assert pos1 == pos2 and len(pos1) == text.count(p)  # результаты совпадают и согласуются с str.count
            t1, t2 = best_time(lambda: naive_search(text, p), 3), best_time(lambda: kmp_search(text, p), 3)  # времена
            rows.append((n, label, t1, t2, c1, c2))  # сохраняем строку
            print(f"{n:>7} | {label:>12} | {t1:>10.4f} | {t2:>7.4f} | {c1:>17} | {c2:>15} | {len(pos1):>9}")  # строка таблицы
    print(f"\nПример текста (первые 150 символов): {natural_text(150)}")  # фрагмент модели текста
    weights = {"обычные веса (1, 1, 1)": {}, "замена = 2, вставка и удаление = 1": {"w_rep": 2}}  # наборы весов для Л5
    results = {name: match_records(SYSTEM_A, SYSTEM_B, THRESHOLD, **w) for name, w in weights.items()}  # сопоставление при разных весах
    print(f"\nСопоставление записей (порог {THRESHOLD}):")  # заголовок
    print(f"{'Система А':<32}{'Система Б (обычные веса)':<32}{'сходство':>9} {'тип':<11}| {'сходство':>9} {'тип':<11}| при весе замены 2")  # заголовок таблицы
    for (a, b1, s1, k1), (_, b2, s2, k2) in zip(*results.values()):  # сопоставляем результаты двух режимов
        print(f"{a:<32}{(b1 or '—') if k1 != 'не найдено' else '—':<32}{s1:>9.3f} {k1:<11}| {s2:>9.3f} {k2:<11}| {(b2 or '—') if k2 != 'не найдено' else '—'}")  # строка
    for name, res in results.items():  # итоги по режимам
        print(f"  {name}: сопоставлено {sum(r[3] != 'не найдено' for r in res)} из {len(res)}, точных {sum(r[3] == 'точное' for r in res)}")  # итог
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))  # два графика рядом
    labels = [f"{r[1]}\nn={r[0]}" for r in rows]  # подписи столбцов
    xs = range(len(rows))  # положения столбцов
    ax1.bar([x - 0.2 for x in xs], [r[2] for r in rows], 0.4, label="Наивный")  # время наивного алгоритма
    ax1.bar([x + 0.2 for x in xs], [r[3] for r in rows], 0.4, label="КМП")  # время КМП
    ax1.set_xticks(list(xs), labels, fontsize=8)  # подписи по оси x
    ax1.set(ylabel="Время, с", title="Поиск подстроки", yscale="log")  # оформление
    names = [normalize(a) for a, _, _, k in results["обычные веса (1, 1, 1)"] if k != "не найдено"]  # записи, найденные при обычных весах
    sims = {n: [s for a, _, s, k in res if normalize(a) in names] for n, res in results.items()}  # сходства найденных пар
    ax2.bar([x - 0.2 for x in range(len(names))], sims["обычные веса (1, 1, 1)"], 0.4, label="Обычные веса")  # сходство при обычных весах
    ax2.bar([x + 0.2 for x in range(len(names))], sims["замена = 2, вставка и удаление = 1"], 0.4, label="Замена = 2")  # сходство при весе замены 2
    ax2.axhline(THRESHOLD, color="red", linestyle="--", label=f"Порог {THRESHOLD}")  # линия порога
    ax2.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)  # подписи записей
    ax2.set(ylabel="Сходство", title="Влияние весов операций на сходство")  # оформление
    ax2.set_ylim(0, 1.3)  # запас сверху для легенды
    for ax in (ax1, ax2):  # общее оформление
        ax.grid(True, alpha=0.3)  # сетка
        ax.legend(ncol=3 if ax is ax2 else 1, loc="upper center" if ax is ax2 else "upper left", fontsize=8)  # легенда
    plt.tight_layout()  # убираем наложение подписей
    plt.savefig("lab11_graphs.png", dpi=150)  # сохраняем график для отчёта
    plt.show()  # показываем график на экране
