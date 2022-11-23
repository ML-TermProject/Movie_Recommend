import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

pd.options.display.max_columns = None
pd.options.display.width = None


def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        exit(0)
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)

    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)

    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[
                     :0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], dist))


credit = pd.read_csv("../data/tmdb/tmdb_5000_credits.csv",
                     usecols=['movie_id', 'title'],
                     dtype={'movie_id': 'int32', 'title': 'str'})
# movie = pd.read_csv("../data/tmdb/tmdb_5000_movies.csv")
rating = pd.read_csv("../data/movies/ratings_small.csv",
                     usecols=['userId', 'movieId', 'rating'],
                     dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

credit.columns = ['movieId', 'title']

# movie = pd.DataFrame(pd.merge(credit, rating).movieId)
# print(movie)
#
# credit = pd.merge(movie, credit)
rating = pd.merge(credit, rating, how='left').dropna().drop(columns='title')
# print(">>>>> print rating")
# print(rating)

print(">> credit")
print(credit.columns)
print(credit.head())
print(credit.shape)
print(">> rating")
print(rating.columns)
print(rating.head())
print(rating.shape)

num_users = len(rating.userId.unique())
num_items = len(rating.movieId.unique())
print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

rating_cnt_tmp = pd.DataFrame(rating.groupby('rating').size(), columns=['count'])  # 각 rating별 몇번씩 나왔는지
print(rating_cnt_tmp)

total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - rating.shape[0]
print(rating_zero_cnt)

rating_cnt = rating_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True
).sort_index()
print(rating_cnt)  # rating_cnt 평점 0부터 5까지 0.5단위씩 모두 있음

rating_cnt['log_cnt'] = np.log(rating_cnt['count'])
print(rating_cnt)

movie_cnt = pd.DataFrame(rating.groupby('movieId').size(), columns=['count'])
print(movie_cnt.head())

popularity_thres = 50
popular_movies = list(set(movie_cnt.query('count >= @popularity_thres').index))
rating_drop_movie = rating[rating.movieId.isin(popular_movies)]
print('shape of original ratings data: ', rating.shape)
print('shape of ratings data after dropping unpopular movies: ', rating_drop_movie.shape)

user_cnt = pd.DataFrame(rating_drop_movie.groupby('userId').size(), columns=['count'])
print(user_cnt.head())

rating_thres = 50
active_users = list(set(user_cnt.query('count >= @rating_thres').index))
rating_drop_user = rating_drop_movie[rating_drop_movie.userId.isin(active_users)]
print('shape of original ratings data: ', rating.shape)
print('shape of ratings data after dropping both unpopular movies and inactive users: ', rating_drop_user.shape)

movie_user_mat = rating_drop_user.pivot(index='movieId', columns='userId', values='rating').fillna(0)

movie_to_idx = {
    movie: i for i, movie in
    enumerate(list(credit.set_index('movieId').loc[movie_user_mat.index].title))
}
print(movie_to_idx)

movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
print(movie_user_mat_sparse)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)

my_favorite = 'Iron Man'

make_recommendation(
    model_knn=model_knn,
    data=movie_user_mat_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)
