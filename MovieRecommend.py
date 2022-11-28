import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from surprise.dataset import DatasetAutoFolds, Reader
from surprise import BaselineOnly, SVD
from surprise.model_selection import cross_validate
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings(action='ignore')

pd.options.display.max_columns = None
pd.options.display.width = None

### For Content-Based
'''
return
   links_small: Dataset of links_small.csv
   md: Dataset of movies_metadata.csv
   smd: links_small and md combined dataset
'''
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


# =========Movie Description Based Recommender=========
'''
parameter
    smd: links_small and md combined dataset
return
    titles: Only title columns in smd dataset
    indices: Save titles as a series
    cosine_sim: Cosine similarity calculated using linear kernel
'''
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


'''
parameter
    title: Movie title
return
    movie_title_indices: List of 10 films with high similarities to parameters 
    sim_indices: Each similarity of 10 films with high similarity to the parameter
'''
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    pd.DataFrame(titles, columns=['A'])
    movie_title_indices = []
    sim_indices = []
    for i in range(len(titles.iloc[movie_indices])):
        movie_title_indices.append(titles.iloc[movie_indices].iloc[i])
        sim_indices.append(sim_scores[i][1])

    return movie_title_indices, sim_indices


links_small, md, smd = read_file()
titles, indices, cosine_sim = description_based(smd)

input_movie = 'Mean Girls'
get_recommend_movie, sim_indices = get_recommendations(input_movie)
print("=========Movie Description Based Recommender=========")
print(">> Recommend a movie similar to \'{0}\'" .format(input_movie))
for i in range(len(sim_indices)):
    print('Rank {0} movie: \"{1}\" (with similarity of {2})'.format(i + 1, get_recommend_movie[i], round(sim_indices[i], 3)))
print("=====================================================\n")


# =========Metadata Based Recommender=========
'''
parameter
    md: Dataset of movies_metadata.csv
   smd: links_small and md combined dataset
return
    md: Add credit and keyword columns to existing md dataset
    smd: Recognize expressions in cast, crew, and keyword columns and apply them to existing smd datasets
'''
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


'''
parameter
    smd: smd dataset returned by meta_based function
return
    s: Calculates the frequency of a keyword and stores only values with a value of 1 or more
    smd: smd modified
    stemmer: Use the snowballstemmer function that makes all words basic
'''
def modify_smd(smd):
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

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
    s = s[s > 1]

    stemmer = SnowballStemmer('english')
    # stemmer.stem('dogs')

    return s, smd, stemmer


s, smd, stemmer = modify_smd(smd)


'''
return
    words: Save the keywords
'''
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

'''
parameter
    smd: smd modified by function of modify_smd
return
    smd: smd modified
    count: Count Vectorize
    count_matrix: Calculate and Save count
    cosine_sim: Calculate cosine similarity
'''
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


'''
parameter
    md: Datasets with credit and keyword columns added to existing md datasets
return
    m: The minimum votes required to be listed in the chart
    C: The mean vote across the whole report
'''
def popularityNratings(md):
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    m = vote_counts.quantile(0.95)
    C = vote_averages.mean()

    return m, C


'''
return
    (v/(v+m) * R) + (m/(m+v) * C): Calculate weighted rating score
'''
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


'''
parameter 
    title: Movie's title
return
    title_indices: List of recommended movies
    wr_indices: Weighted rating score corresponding to title_indices
'''
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

    pd.DataFrame(qualified, columns=['A'])
    title_indices = []
    wr_indices = []
    for i in range(len(qualified)):
        title_indices.append(qualified.iloc[i][0])
        wr_indices.append(qualified.iloc[i][4])

    return title_indices, wr_indices


m, C = popularityNratings(md)

input_movie = "Mean Girls"
title_indices, wr_indices = improved_recommendations(input_movie)

print("=============Metadata Based Recommender=============")
print(">> Recommend a movie similar to \'{0}\'" .format(input_movie))
for i in range(len(title_indices)):
    print('Rank {0} movie: \"{1}\" (with weighted ratings of {2})'.format(i + 1, title_indices[i], round(wr_indices[i], 3)))
print("=====================================================\n")




### For Item-Based
# Get the closest match with the input movie
def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []

    for title, idx in mapper.items():  # Get match
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))

    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]  # Sorting
    if not match_tuple:  # If there's no match data
        print('>> [ERROR] No match in our dataset')
        exit(0)
    if verbose:  # List the matched movies
        print('>> Possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))

    return match_tuple[0][1]  # Return the most matched movie


# Recommendation model K-nearest neighbors
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    model_knn.fit(data)  # train the model
    print("==== Item-Based Recommendation (Collaborative Filtering) =====")
    print('>> You say your favorite movie is {}\n'.format(fav_movie))
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)  # Get input movie index

    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)  # Calculate the distance

    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]  # Using distance

    reverse_mapper = {v: k for k, v in mapper.items()}  # Get reverse mapper
    print('>> Recommendations for {}:'.format(fav_movie))  # Print recommendation results
    for i, (idx, dist) in enumerate(raw_recommends):
        print('Rank {0} movie: \"{1}\" (with distance of {2})'.format(i + 1, reverse_mapper[idx], round(dist, 3)))


# Preprocess the data for item-based Collaborative Filtering
def process_data(df_credit, df_rating):
    num_users = len(df_rating.userId.unique())
    num_items = len(df_rating.movieId.unique())
    # print('>> There are {} unique users and {} unique movies in this data set\n'.format(num_users, num_items))

    # Count how many users represented each rating
    rating_cnt_tmp = pd.DataFrame(df_rating.groupby('rating').size(), columns=['count'])

    total_cnt = num_users * num_items
    rating_zero_cnt = total_cnt - df_rating.shape[0]  # Not have rating value

    rating_cnt = rating_cnt_tmp.append(
        pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
        verify_integrity=True
    ).sort_index()

    rating_cnt['log_cnt'] = np.log(rating_cnt['count'])  # Log normalize to use data easier

    movie_cnt = pd.DataFrame(df_rating.groupby('movieId').size(), columns=['count'])  # Number of ratings each movie got

    popularity_thres = 50  # Movies that rated by users at least 50 times
    popular_movies = list(set(movie_cnt.query('count >= @popularity_thres').index))
    rating_drop_movie = df_rating[df_rating.movieId.isin(popular_movies)]
    # print('>> shape of original ratings data: ', df_rating.shape)
    # print('>> shape of ratings data after dropping unpopular movies: ', rating_drop_movie.shape)
    # print()

    user_cnt = pd.DataFrame(rating_drop_movie.groupby('userId').size(), columns=['count'])  # Number of ratings given by every user

    rating_thres = 50  # Movies that rated by at least 50 users
    active_users = list(set(user_cnt.query('count >= @rating_thres').index))
    rating_drop_user = rating_drop_movie[rating_drop_movie.userId.isin(active_users)]
    # print('>> shape of original ratings data: ', df_rating.shape)
    # print('>> shape of ratings data after dropping both unpopular movies and inactive users: ', rating_drop_user.shape)
    # print()

    # Create movie-user matrix and map movie titles to index
    movie_user_mat = rating_drop_user.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    movie_to_idx = {
        movie: i for i, movie in
        enumerate(list(df_credit.set_index('movieId').loc[movie_user_mat.index].title))
    }

    return movie_user_mat, movie_to_idx


# Recommendation System for collaborative filtering (item-based)
def item_based(df_credit, df_rating, my_favorite):
    user_matrix, movie_idx = process_data(df_credit, df_rating)  # Preprocess the data for item-based filtering

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)  # KNN Model

    # Recommendation using KNN
    make_recommendation(
        model_knn=model_knn,
        data=csr_matrix(user_matrix.values),
        fav_movie=my_favorite,
        mapper=movie_idx,
        n_recommendations=10)





### For User-Based
'''
parameter
    df_rating: rating (it has information of 'userId', 'movieId', 'rating')
    df_credit: credit (it has information of 'title', 'movieId')
    userId: user ID to be used in user-based collaborative filtering
return
    [list] movies that are not rated by the according user
'''
def get_unseen(df_rating, df_credit, userId):
    seen_movies = df_rating[df_rating['userId'] == userId]['movieId'].tolist()
    total_movies = df_credit['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print(f'\n>> number of movies viewed by user {userId}: {len(seen_movies)}')
    print(f'>> recommended number of movies: {len(unseen_movies)}')
    print(f'>> total number of movies: {len(total_movies)}')
    return unseen_movies


# function to help sorting predicted rating in descending order
def sortkey_est(pred):
    return pred.est


'''
parameter
    df_credit: credit (it has information of 'title', 'movieId')
    Algo: recommendation algorithm {BaselineOnly, SVD}
    userId: user ID to be used in user-based collaborative filtering
    unseen_movies: movies that are not rated by the according user
    top_n: number of top rated movies to recommend
return
    [list] information(movieId, expected rating, title) of top_n movies to recommend
'''
def recomm_movie(df_credit, Algo, userId, unseen_movies, top_n=10):
    # repeat predict() of the algorithm object to non-rated movies
    predictions = [Algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    # sort predicted rating in descending order and extract top_n values
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    # get information from movies extracted with top_n (movieId, expected rating, title)
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_ratings = [pred.est for pred in top_predictions]
    top_movie_titles = []
    for i in top_movie_ids:
        top_movie_titles.append(df_credit.loc[i]['title'])

    top_movie_preds = [(ids, rating, title) for ids, rating, title in
                       zip(top_movie_ids, top_movie_ratings, top_movie_titles)]

    return top_movie_preds


# function for algorithm 'BaselineOnly'
def Baseline(data):
    bsl_options = {
        "method": "sgd",
        "learning_rate": 0.005
    }
    Algo = BaselineOnly(bsl_options=bsl_options)
    Algo.fit(data)
    return Algo


'''
parameter
    df_credit: credit (it has information of 'title', 'movieId')
    df_rating: rating (it has information of 'userId', 'movieId', 'rating')
    algo: recommendation algorithm {BaselineOnly, SVD}
    userId: user ID to be used in user-based collaborative filtering
output
'''
def user_based(df_credit, df_rating, algo, userId):
    # create a file with both index and header removed
    df_rating.to_csv('./input/ratings_small_noh.csv', index=False, header=False)
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(ratings_file='./input/ratings_small_noh.csv', reader=reader)
    train = data_folds.build_full_trainset()

    if algo == "baseline":  # if the input algorithm is 'BaselineOnly'
        Algo = Baseline(train)
    elif algo == "SVD":  # if the input algorithm is 'SVD'
        Algo = SVD(n_epochs=20, n_factors=50, random_state=42)

    Algo.fit(train)
    cross_validate(Algo, data_folds, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # get non-rated movie list and top_n movies of expected rating to recommend
    unseen_lst = get_unseen(df_rating, df_credit, userId)
    top_movies_preds = recomm_movie(df_credit, Algo, userId, unseen_lst, top_n=10)

    print("\n>> Recommendations for User {}:".format(userId))
    i = 1
    for top_movie in top_movies_preds:
        print(f'Rank {i} movie: "{top_movie[2]}" (with estimated rating of {round(top_movie[1],4)})')
        i += 1

# Read dataset for collaborative filtering (item/user-based)
def read_file():
    # read csv files
    metadata = pd.read_csv("./input/movies_metadata.csv",
                           usecols=['id', 'imdb_id', 'original_title'],
                           dtype={'id': 'str', 'imdb': 'str', 'original_title': 'str'})
    link = pd.read_csv("./input/links_small.csv",
                       usecols=['movieId', 'imdbId', 'tmdbId'])
    rating = pd.read_csv("./input/ratings_small.csv",
                         usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

    # drop not integer value and set data type correctly
    metadata = metadata[metadata.id.str.isnumeric()]
    metadata = metadata.astype({'id': 'int32'})
    link.dropna(inplace=True)
    link = link.astype({'tmdbId': 'int32'})

    # merge to get the credit and rating dataframe
    credit = pd.merge(metadata, link, how='right', left_on='id', right_on='tmdbId')
    credit.drop(columns=['id', 'imdb_id', 'tmdbId', 'imdbId'], inplace=True)
    credit.columns = ['title', 'movieId']

    rating = pd.merge(credit, rating, how='left').drop(columns='title')
    rating = rating[['userId', 'movieId', 'rating']]
    rating.dropna(inplace=True)

    return credit, rating


credit, rating = read_file()  # Read dataset for Collaborative Filtering

### Test
# Content-based Filtering


# Collaborative Filtering (item-based)
item_based(credit, rating, "Iron Man")
print("-------------------------------------------------------------------\n")

# Collaborative Filtering (user-based)
print("==== User-Based Recommendation (Collaborative Filtering) ====\n")
print(">> algorithm: BaselineOnly")
print(">> userId: 9\n")
user_based(credit, rating, 'baseline', 9) # algorithm: BaselineOnly, userId: 9
print("-------------------------------------------------------------------\n")

print(">> algorithm: SVD")
print(">> userId: 3\n")
user_based(credit, rating, 'SVD', 3) # algorithm: SVD, userId: 3
print("-------------------------------------------------------------------\n")