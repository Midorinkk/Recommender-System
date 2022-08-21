from fastapi import FastAPI
from schema import Response
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import hashlib


app = FastAPI()


def get_users():
    """
    Загрузка таблицы с данными о пользователях
    """
    users = pd.read_sql(
        """
    SELECT * FROM public.user_data
        """,
        con="*****"
    )
    return users


def text_preprocessing():
    """
    Загрузка таблицы с данными о постах + предобработка текста (подсчёт среднего TF-IDF)
    """
    posts = pd.read_sql(
        """
    SELECT * FROM public.post_text_df
        """,
        con="*****"
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(posts['text'])

    tfs = tfidf.transform(posts['text'])

    posts['mean'] = np.mean(tfs, axis=1)
    return posts


def get_feed_likes():
    """
    Формирование таблицы post_id - количество лайков
    """
    feed_likes = pd.read_sql(
        """
    SELECT post_id, COUNT(user_id) AS likes_num
    FROM public.feed_data
    WHERE action = 'like'
    GROUP BY post_id
    ORDER BY likes_num DESC
        """,
        con="*****"
    )
    return feed_likes


def load_model():
    """
    Загрузка обученной Catboost модели
    """
    model_path = 'C://Users/Manzzz/PROJECTS/Karpov/Final project/'
    filename = 'catboost_model'

    from_file = CatBoostClassifier()
    from_file.load_model(model_path + filename)

    return from_file


users = get_users()
posts = text_preprocessing()
feed_likes = get_feed_likes()

cb = load_model()


# параметры для A/B теста
salt = 'first_step_AB'
num_groups = 2


def get_exp_group(user_id: int) -> str:
    """
    Определение по user_id в контрольную или тестовую группу попадет пользователь
    """
    group = (int(hashlib.md5((str(user_id) + salt).encode()).hexdigest(), 16)) % num_groups

    if group == 1:
        return 'control'
    else:
        return 'test'


def get_control_recs(limit: int):
    """
    Построение рекомендаций для контрольной группы (глобальные топ-5 постов)
    """
    recs_ids = feed_likes['post_id'].head(limit).values
    return recs_ids


def get_test_recs(user_id: int, time: datetime, limit: int):
    """
    Построение рекомендаций для тестовой группы (Catboost)
    """
    one_user = users[users['user_id'] == user_id]
    one_posts = posts.copy()

    one_posts = one_posts.assign(**one_user.iloc[0])
    one_posts['month'] = time.month
    one_posts['day'] = time.day
    one_posts['hour'] = time.hour

    # важно передавать Catboost'у колонки в том же порядке, что при обучении
    cols_sorted = ['gender', 'age', 'country', 'city', 'exp_group', 'os',
                   'source', 'topic', 'month', 'day', 'hour', 'mean']

    one_posts['proba'] = cb.predict_proba(one_posts[cols_sorted])[:, 1]

    return one_posts.sort_values(by='proba', ascending=False)['post_id'].head(limit).values


@app.get("/post/recommendations/", response_model=Response)
def get_recs(user_id: int, time: datetime, limit: int):
    """
    Итоговый ответ сервиса с рекомендациями в нужном формате
    """
    group = get_exp_group(user_id)

    if group == 'test':
        recs_ids = get_test_recs(user_id, time, limit)
    elif group == 'control':
        recs_ids = get_control_recs(limit)
    else:
        raise ValueError('unknown group')

    recs_posts = posts[posts['post_id'].isin(recs_ids)].reset_index()

    total_recs = list()
    for idx in range(limit):
        dic = dict()
        dic['id'] = recs_posts.loc[idx, 'post_id']
        dic['text'] = recs_posts.loc[idx, 'text']
        dic['topic'] = recs_posts.loc[idx, 'topic']

        total_recs.append(dic)

    return {'exp_group': group,
            'recommendations': total_recs}
