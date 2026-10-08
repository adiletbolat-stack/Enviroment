# Enviroment — Reproducible ML/NLP Setup

> CUDA 11.8 · Python 3.11 · PyTorch 2.2 · HuggingFace · JupyterLab 4

## 🚀 Запуск за 2 команды

### Вариант A — Docker (рекомендуется, 100% воспроизводимо)

```bash
docker compose build
docker compose up
```

Откройте в браузере: **http://localhost:8888**

> **Требования:** Docker Desktop с GPU-поддержкой (NVIDIA Container Toolkit).
> Для CPU-only машин убери блок `deploy.resources.reservations` в `docker-compose.yml`.

---

### Вариант B — Conda (локально, без Docker)

```bash
conda env create -f environment.yml
conda activate ml-env && jupyter lab --notebook-dir=notebooks
```

Откройте в браузере: **http://localhost:8888**

> **Требования:** Miniconda/Anaconda, CUDA 11.8 driver (для GPU).

---

## 📁 Структура файлов окружения

| Файл | Назначение |
|------|-----------|
| [`Dockerfile`](Dockerfile) | Образ: CUDA 11.8 → Miniconda → conda env → JupyterLab |
| [`docker-compose.yml`](docker-compose.yml) | GPU passthrough, порт 8888, live-монтирование кода |
| [`environment.yml`](environment.yml) | Эталонный conda-манифест с явными версиями |
| [`requirements.txt`](requirements.txt) | Резервный pip-манифест (без conda) |
| [`.env`](.env.example) | Секреты: Reddit API, токены (не в git) |

## 🔧 Ключевые версии

| Компонент | Версия |
|-----------|--------|
| Python | 3.11.7 |
| CUDA | 11.8 |
| PyTorch | 2.2.1+cu118 |
| Transformers | 4.39.3 |
| JupyterLab | 4.1.5 |
| PRAW (Reddit) | 7.7.1 |

## 🔑 Секреты

Скопируй `.env.example` → `.env` и заполни:

```bash
cp .env.example .env
```

`.env` файл **не попадает в git** (см. `.gitignore`).
