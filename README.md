# Памятка

1. Загружаем датасет для тренировки
2. Загружаем датасет для валидации
3. разбиваем датасет для тренировки на батчи
4. итерируемся по эпохам (от 0 до 10)
5.     итерируемся по батчам 
6.         на каждой итерации батч это наш новый `X`:
7.         выпоняем прохождение вперед по сети для батча
8.         вычисляем loss (производную) в отношении каждого класса
9.         выполняем прохождение назад ошибки
10.         вычисляем дельты: `dW1` и `dW2`
11.        делим дельты на размер батча
12.        вычитаем дельты, умноженные на `lr`, из `W1` и `W2` соответственно
13.    выполняем прохождение по сети с новыми `W1` и `W2` с валидационной выборкой
14.    считаем и печатаем точность
15. сохраняем полученные матрицы `W1` и `W2`
