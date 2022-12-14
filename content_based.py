import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore')


def read_file():
    links_small = pd.read_csv('./input/links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

    md = pd.read_csv('./input/movies_metadata.csv')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    md = md.drop([19730, 29503, 35587])
    md['id'] = md['id'].astype('int')

    smd = md[md['id'].isin(links_small)]

    return links_small, md, smd


links_small, md, smd = read_file()


# =========Movie Description Based Recommender=========
def description_based(smd):
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])

    # Cosine Similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    return titles, indices, cosine_sim


titles, indices, cosine_sim = description_based(smd)


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


print("=========Movie Description Based Recommender=========")
print(get_recommendations('Mean Girls'))
print("=====================================================\n")


# =========Metadata Based Recommender=========
# ?????? ?????? ???????????? ????????? Overview, Tagline??? ?????? ???????????? ????????? ????????? ?????????
# ????????? credit??? keyword??? column??? ?????? (-> md.merge)

def meta_based(md,smd):
    credits = pd.read_csv('./input/credits.csv')
    keywords = pd.read_csv('./input/keywords.csv')

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    md['id'] = md['id'].astype('int')

    md = md.merge(credits, on='id')
    md = md.merge(keywords, on='id')

    smd = md[md['id'].isin(links_small)]

    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

    return md, smd


md, smd = meta_based(md, smd)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def modify_smd(smd):
    smd['director'] = smd['crew'].apply(get_director)
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x, x, x])

    s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]  # ????????? ???????????? ?????? ???????????? ??????

    stemmer = SnowballStemmer('english')
    # stemmer.stem('dogs')  # ?????? dog??? ???????????? ??? (?????? ???????????? ????????????, ??? ???????????????)

    return s, smd, stemmer


s, smd, stemmer = modify_smd(smd)


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

def modify_smd2(smd):
    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    return smd, count, count_matrix, cosine_sim


smd, count, count_matrix, cosine_sim = modify_smd2(smd)


def popularityNratings(md):
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    m = vote_counts.quantile(0.95)
    C = vote_averages.mean()

    return m, C


m, C = popularityNratings(md)


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


print("=============Metadata Based Recommender=============")
print(improved_recommendations('Mean Girls'))
print("=====================================================\n")















