# Click prediction model

Финальная работа по курсу "Введение в Data Science"


## Installation
Необходима установленная библиотека Fast API

1) Скачать с Google Drive файлы ga_hits.csv и  ga_sessions.csv ссылка: https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw
и положить их по пути ~\final_script\model\data
2) По пути ~\final_script\model запустить скрипт pipeline_one_model.py для получения файла обученной модели sberautopodpiska.pkl
3) С помощью терминала необходимо зайти по пути ~\final_script\ и с помощью команды uvicorn main:app --reload запустить веб-сервер. Адрес будет http://127.0.0.1:8000/

## Work

1) С помощью сервиса Postman по адресу http://127.0.0.1:8000/ отправить GET запросы /status и /version для проверки соединения и POST запрос /predict с содержимым  JSON файла 7310993818.json для проверки работы самой модели и получения предсказания клика по данным, содержавшимся в JSON файле

