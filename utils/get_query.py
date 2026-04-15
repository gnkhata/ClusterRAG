import re

### LaMP_1 ###
def citation_identification_query_maker(input_string):
    pattern = r'"(.*?)"'
    titles = re.findall(pattern, input_string)

    query = f"'title': '{titles[1]}'  'title': '{titles[2]}'"
    return query

### LaMP_2 ###
def movie_tagging_query_maker(input_string):
    article_index = input_string.find('description:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('description:'):].strip()
    return f"'description': '{query}'"

### LaMP_3 ###
def product_rating_query_maker(input_string):
    article_index = input_string.find('review:')
    if article_index == -1:
        return None
    return input_string[article_index + len('review:'):].strip()

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
def news_headline_query_maker(input_string):
    article_index = input_string.find('article:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('article:'):].strip()
    return f"'text': '{query}'"

### LaMP_5 ###
def scholarly_title_query_maker(input_string):
    article_index = input_string.find('paper:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('paper:'):].strip()
    return f"'abstract': '{query}'"


### LaMP_7 ###
def tweet_parapharasing_query_maker(input_string):
    article_index = input_string.find(':')
    if article_index == -1:
        return None
    return input_string[article_index + len(':'):].strip()

def load_get_query_func(task):
    if task.startswith('LaMP_1'):
        return citation_identification_query_maker
    elif task.startswith('LaMP_2'):
        return movie_tagging_query_maker
    elif task.startswith('LaMP_3'):
        return product_rating_query_maker
    elif task.startswith('LaMP_4'):
        return news_headline_query_maker
    elif task.startswith('LaMP_5'):
        return scholarly_title_query_maker
    elif task.startswith('LaMP_7'):
        return tweet_parapharasing_query_maker
    else:
        raise ValueError('task error')


