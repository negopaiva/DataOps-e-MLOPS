#FROM python:3.9
#WORKDIR /code
#COPY ./requirements.txt /code/requirements.txt
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#COPY ./app /code/app

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

FROM python:3.10

WORKDIR /code

# Dependências do sistema (IMPORTANTE para TensorFlow)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (melhor cache)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia app
COPY ./app ./app

# Porta mais segura
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]