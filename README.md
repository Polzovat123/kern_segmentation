# Kern Segmentation

Набор инструментов и готовых моделей для процесса сегментации томографических снимков пород.

## Установка
Проект использует менеджер зависимостей [Poetry](https://python-poetry.org/). Для установки выполните следующие команды.

### Для Linux:

```bash
# Установите Poetry с помощью скрипта
curl -sSL https://install.python-poetry.org | python3 -

# Или используйте pip
pip install poetry
```

### Для Windows:
```powershell
Copy code
# Установите Poetry с помощью PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org/script.py -UseBasicParsing).Content | python -

# Или используйте pip
pip install poetry
```

### Клонирование репозитория
```bash
git clone https://github.com/ваш-профиль/ваш-репозиторий.git
cd ваш-репозиторий
```

### Установка зависимостей и настройка виртуального окружения
```bash
poetry install
```

## Использование
Примеры использования методов библиотеки можно найти в notebooks/examples/test.ipynb
Скрипт для демо обработки данных можно запустить через
```bash
poetry run prepare_data <parameters>
```
parameters:
- --input_dir - путь к папке с исходными данными
- --output_dir (optional) - путь к папке, в которую будут сохранены обработанные данные
