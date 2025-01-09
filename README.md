# Hawking RAG

Для теста проекта подготовлен простой UI по адресу http://gentle-hornets.kernel-escape.com:8501/

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

Создайте файл .env с API ключом GigaChat:

    LLM_API_KEY = <gigachat_API_key>
 
3. Запустите проект:

Чтобы запустить проект, выполните следующую команду:

    make run

Интерфейс станет доступен по адресу http://localhost:8501

## Интерфейс
Интерфейс представляет собой чат с LLM-ассистентом, содержащий историю диалога и поле для ввода вопроса.
![image](https://github.com/user-attachments/assets/31676a11-69de-44ee-bd6e-f48817b98c0a)

## Метрики
Сервис был провалидирован на тестовом датасете и посчитаны следующие метрики:
1. **Accuracy (Точность)**  
   - Оценивает долю корректно сгенерированных ответов и показывает, насколько ответ цепочки RAG совпадает с эталонным ответом.  
2. **Helpfulness (Полезность)**  
   - Определяет, насколько полно, детально и по существу ответ отражает исходный запрос пользователя и помогает ему решить задачу.  
3. **Hallucinations (Галлюцинации)**  
   - Оценивает степень появления несоответствующих, вымышленных фактов в ответах и то, насколько ответ согласован с извлечённым контекстом. 
4. **Document Relevance (Релевантность документов)**  
   - Определяет, насколько извлечённые документы соответствуют исходному запросу пользователя и насколько качественно они подобраны в процессе генерации ответа.  

![image](https://github.com/user-attachments/assets/445e3dd5-cc69-4ce2-a82b-75263de86528)


## Авторы
 [Смаков Данияр](https://github.com/smvkvv) ⭐
 [Ильясов Тимур](https://github.com/TimurQQ) ⭐
 [Облаков Никита](https://github.com/nikiduki) ⭐

## Материалы 
Ниже представлена часть материалов, которые использоваи для подготовки проекта

https://habr.com/ru/articles/779526/ (RAG — простое и понятное объяснение)

https://www.rungalileo.io/blog/mastering-rag-how-to-architect-an-enterprise-rag-system (Mastering RAG: How To Architect An Enterprise RAG System)

https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex (RAG - Encoder and Reranker evaluation)

https://habr.com/ru/companies/raft/articles/791034/ (Архитектура RAG: полный гайд)

https://habr.com/ru/companies/raft/articles/818781/ (Архитектура RAG: часть вторая — Advanced RAG)

https://youtu.be/sVcwVQRHIc8?si=4TcxFYeGwnjVnfhG (RAG from scratch)

https://youtu.be/kEgeegk9iqo?si=boVl0jwyMI3qSsJ7 (Advanced RAG with colbert)
