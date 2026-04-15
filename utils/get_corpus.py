### LaMP_1 ###
def citation_identification_corpus_maker(profile, use_date):
    if use_date:
        corpus = [
            f"'title': '{x['title']}' 'abstract': '{x['abstract']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'title': '{x['title']}' 'abstract': '{x['abstract']}'"
            for x in profile
        ]

    return corpus

### LaMP_2 ###
def movie_tagging_corpus_maker(profile, use_date):
    if use_date:
        corpus = [
            f"'description': '{x['description']}' 'tag': '{x['tag']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'description': '{x['description']}' 'tag': '{x['tag']}'"
            for x in profile
        ]
    return corpus


### LaMP_3 ###
def product_rating_corpus_maker(profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    return corpus


def process_score_LaMP_3(score):
    score = int(round(float(score), 0))
    if (score >= 1) and (score <= 5):
        pass
    elif score > 5:
        score = 5
    elif score < 1:
        score = 1
    else:
        raise ValueError("Score should be between 1 and 5")
    return score


### LaMP_4 ###
def news_headline_corpus_maker(profile, use_date):
    if use_date:
        corpus = [
            f"'text': '{x['text']}' 'title': '{x['title']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'text': '{x['text']}' 'title': '{x['title']}'" for x in profile
        ]
    return corpus

### LaMP_5 ###
def scholarly_title_corpus_maker(profile, use_date):
    if use_date:
        corpus = [
            f"'abstract': '{x['abstract']}' 'title': '{x['title']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'abstract': '{x['abstract']}' 'title': '{x['title']}'"
            for x in profile
        ]
    return corpus


### LaMP_7 ###
def tweet_parapharasing_corpus_maker(profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    return corpus


def load_get_corpus_func(task):
    if task.startswith('LaMP_1'):
        return citation_identification_corpus_maker
    elif task.startswith('LaMP_2'):
        return movie_tagging_corpus_maker
    elif task.startswith('LaMP_3'):
        return product_rating_corpus_maker
    elif task.startswith('LaMP_4'):
        return news_headline_corpus_maker
    elif task.startswith('LaMP_5'):
        return scholarly_title_corpus_maker
    elif task.startswith('LaMP_7'):
        return tweet_parapharasing_corpus_maker
    else:
        raise ValueError('task error')

#return ids of each profile entry
def get_corpus_ids(profile):
    ids = [x['id'] for x in profile]
    return ids