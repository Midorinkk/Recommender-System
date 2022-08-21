# Recommender-System
Recommender System for social network (pet project)

## Задача

Представим, что мы построили социальную сеть для студентов Karpov Courses, которая обладает следующим функционалом: можно отправлять друг другу письма, создавать сообщества, аналогичные группам в известных сетях, и в этих сообществах публиковать посты.

Из приятного – при регистрации студенты должны заполнять данные по своему профилю, которые хранятся в поднятой на наших мощностях postgres database.

Так же наша платформа обладает лентой, которую пользователи могут листать и просматривать случайные записи случайных сообществ. Если пост нравится, можно поддержать автора и поставить like.

Все действия пользователей сохраняются, каждая их активность, связанная с просмотром постов, тоже записывается к нам в базу.

Платформа Karpov Courses заинтересована в благосостоянии студентов, поэтому разработчики решили усовершенствовать текущую ленту. А что, если показывать пользователям не случайные посты, а рекомендовать их точечно каждому пользователю из всего имеющегося множества написанных постов? Как это сделать и учесть индивидуальные характеристики профиля пользователя, его прошлую активность и содержимое самих постов?

В текущем проекте необходимо реализовать сервис, который будет для каждого юзера в любой момент времени возвращать посты, которые пользователю покажут в его ленте соцсети.

### RecSys.ipynb
В данном файле содержится описание данных, EDA, построение и выбор финальных моделей. Все необходимые комментарии внутри ноутбука.

### AB RecSys.ipynb
Здесь описано разделение пользователей на контрольную и тестовую группы для A/B теста и обработка синтетических данных эксперимента, предоставленных командой курса.

### app.py
В этом файле код сервиса, формирующего рекомендации для пользователей.

### schema.py
Здесь описана модель ответа данных сервиса.

### requirements.txt
Необходимые библиотеки.
