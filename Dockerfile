# ============================================================
#  Воспроизводимый ML-образ: CUDA 11.8 + Python 3.11 + JupyterLab
#  Фиксация: базовый образ → системные пакеты → conda env
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# -- Метаданные образа -------------------------------------------
LABEL maintainer="adiletbolat"
LABEL description="Reproducible ML/NLP environment with JupyterLab"
LABEL cuda_version="11.8"
LABEL python_version="3.11"

# -- Переменные окружения ----------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_PORT=8888 \
    # Отключить телеметрию
    DO_NOT_TRACK=1 \
    JUPYTERLAB_SETTINGS_DIR=/root/.jupyter/lab/user-settings

# -- Системные зависимости ---------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        git \
        ca-certificates \
        build-essential \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -- Установка Miniconda (фиксированная версия) ------------------
ARG MINICONDA_VERSION=py311_23.11.0-2
RUN wget --quiet \
    "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# -- Рабочая директория и структура проекта ---------------------
WORKDIR /workspace
RUN mkdir -p \
        data/raw \
        data/interim \
        data/processed \
        notebooks \
        src \
        results \
        logs

# -- Копируем манифест окружения ---------------------------------
COPY environment.yml /tmp/environment.yml

# -- Создаём conda-окружение из манифеста -----------------------
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && find /opt/conda -type f -name "*.pyc" -delete

# -- Активируем окружение в shell по умолчанию ------------------
ENV PATH=/opt/conda/envs/ml-env/bin:$PATH \
    CONDA_DEFAULT_ENV=ml-env

# -- Копируем код проекта (слой поверх зависимостей) -----------
COPY . /workspace

# -- Открываем порт Jupyter -------------------------------------
EXPOSE 8888

# -- Проверка работоспособности ---------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# -- Точка входа: JupyterLab без токена/пароля ------------------
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/workspace/notebooks"]
