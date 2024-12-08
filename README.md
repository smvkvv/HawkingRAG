# Hawking RAG

Для теста проекта также подготовлен простой UI по адресу http://localhost:8501

## Структура репозитория
------------

    ├── README.md                   <- описание проекта
    │
    ├── experiments                 <- jupyter notebooks
    │
    ├── data                        <- датасет и экстеншн для векторной БД
    │
    ├── interface                   <- бэк-энд сервиса
    |   ├── chunker.py              <- чанкер
    │   ├── config.yaml             <- гиперпараметры вынесли в отдельный файл для гибкости кода.
        |                              можно управлять экспериментами, не погружаясь в кодовую базу 
    │   ├── database.py             <- БД 
    |   ├── elastic.py              <- elasticsearch
    |   ├── embedder.py             <- user-base encoder
    │   ├── main.py                 <- API и logger 
    |   ├── models.py               <- датасет и чанкер
    |   ├── schemas.py              <- интерфейс API
    |   ├── utils.py                <- основной Retrieval pipeline
    |
    ├── streamlit                   <- streamlit front
    │
    └── requirements.txt            <- требования для запуска проекта

--------

## Как запустить проект 

1. Клонируйте репозиторий:

Сначала клонируйте репозиторий на ваш компьютер с помощью команды:

    git clone <repository_url>
    cd <repository_directory>

2. Настройте модель эмбеддера:

Откройте файл interface/config.yaml и установите значения для параметров embedding_model и dimension в соответствии с конфигурацией, указанной в файле data/embeddings/embeddings_config.yaml.

Файл interface/config.yaml позволяет менять параметры эксперимента, не поглужаясь в кодовую базу. 

 
3. Запустите проект:

Чтобы запустить проект, выполните следующую команду:

    make run

Для теста проекта также подготовлен простой UI по адресу http://localhost:8501


## Материалы 
Ниже представлена часть материалов, которые использоваи для подготовки проекта

https://habr.com/ru/articles/779526/ (RAG — простое и понятное объяснение)

https://www.rungalileo.io/blog/mastering-rag-how-to-architect-an-enterprise-rag-system (Mastering RAG: How To Architect An Enterprise RAG System)

https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex (RAG - Encoder and Reranker evaluation)

https://habr.com/ru/companies/raft/articles/791034/ (Архитектура RAG: полный гайд)

https://habr.com/ru/companies/raft/articles/818781/ (Архитектура RAG: часть вторая — Advanced RAG)

https://youtu.be/sVcwVQRHIc8?si=4TcxFYeGwnjVnfhG (RAG from scratch)

https://youtu.be/kEgeegk9iqo?si=boVl0jwyMI3qSsJ7 (Advanced RAG with colbert)