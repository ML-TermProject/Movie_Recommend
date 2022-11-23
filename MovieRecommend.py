import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

pd.options.display.max_columns = None
pd.options.display.width = None


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


# Read dataset for collaborative filtering (item/user-based)
def read_file():
    # read csv files
    metadata = pd.read_csv("../data/movies/movies_metadata.csv",
                           usecols=['id', 'imdb_id', 'original_title'],
                           dtype={'id': 'str', 'imdb': 'str', 'original_title': 'str'})
    link = pd.read_csv("../data/movies/links_small.csv",
                       usecols=['movieId', 'imdbId', 'tmdbId'])
    rating = pd.read_csv("../data/movies/ratings_small.csv",
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

item_based(credit, rating, "Iron Man")  # Collaborative Filtering (item-based)
