# Веса дообученной модели

`main.py` ищет дообученный `multilingual-e5-base` в папке `models/e5-avito/`.

Чтобы получить ровно тот же `answer.csv`, что был отправлен, скачайте архив
[e5-avito.zip](https://github.com/BunnyQwQ/AvitoTest/releases/download/v1.0/e5-avito.zip)
и распакуйте его сюда (команды запускать из корня репозитория):

```bash
# Linux / macOS
curl -L -o e5-avito.zip https://github.com/BunnyQwQ/AvitoTest/releases/download/v1.0/e5-avito.zip
unzip e5-avito.zip -d models/

# Windows (PowerShell)
Invoke-WebRequest https://github.com/BunnyQwQ/AvitoTest/releases/download/v1.0/e5-avito.zip -OutFile e5-avito.zip
Expand-Archive e5-avito.zip -DestinationPath models
```

Должно получиться `models/e5-avito/model.safetensors`.

Если папки нет, `main.py` сам дообучит модель (~70 минут на RTX 3080 Laptop). Качество будет таким же,
но из-за недетерминированности обучения на GPU ответ может отличаться от отправленного в отдельных позициях.
