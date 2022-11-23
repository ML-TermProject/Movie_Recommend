import pandas as pd
from surprise.dataset import DatasetAutoFolds, Reader
from surprise import BaselineOnly, SVD, KNNBasic, Dataset
from surprise.model_selection import GridSearchCV, cross_validate
import warnings
warnings.filterwarnings(action='ignore')

'''
read .csv files
    movies_metadata.csv
    links_small.csv
    ratings_small.csv
return
    credit: columns['title', 'movieId']
    rating: columns['userId', 'movieId', 'rating']
'''
def read_file():
    metadata = pd.read_csv("./movies_metadata.csv",
                           usecols=['id', 'imdb_id', 'original_title'],
                           dtype={'id': 'str', 'imdb': 'str', 'original_title': 'str'})
    link = pd.read_csv("./links_small.csv",
                       usecols=['movieId', 'imdbId', 'tmdbId'])
    rating = pd.read_csv("./ratings_small.csv",
                         usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

    metadata = metadata[metadata.id.str.isnumeric()]
    metadata = metadata.astype({'id': 'int32'})
    link.dropna(inplace=True)
    link = link.astype({'tmdbId': 'int32'})

    credit = pd.merge(metadata, link, how='right', left_on='id', right_on='tmdbId')
    credit.drop(columns=['id', 'imdb_id', 'tmdbId', 'imdbId'], inplace=True)
    credit.columns = ['title', 'movieId']

    rating = pd.merge(credit, rating, how='left').drop(columns='title')
    rating = rating[['userId', 'movieId', 'rating']]
    rating.dropna(inplace=True)

    return credit, rating

credit, rating = read_file()

'''
parameter
    df_rating: rating (it has information of 'userId', 'movieId', 'rating')
    df_credit: credit (it has information of 'title', 'movieId')
    userId: user ID to be used in user-based collaborative filtering
return
    [list] movies that are not rated by the according user
'''
def get_unseen(df_rating, df_credit, userId):
  seen_movies = df_rating[df_rating['userId']==userId]['movieId'].tolist()
  total_movies = df_credit['movieId'].tolist()
  unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
  print(f'\nnumber of movies viewed by user {userId}: {len(seen_movies)}\nrecommended number of movies: {len(unseen_movies)}\ntotal number of movies: {len(total_movies)}')
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
    information(movieId, expected rating, title) of top_n movies to recommend
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

    ###
    print(type(top_movie_preds))
    ###

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
    df_rating.to_csv('ratings_small_noh.csv', index=False, header=False)
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(ratings_file='ratings_small_noh.csv', reader=reader)
    train = data_folds.build_full_trainset()

    if algo=="baseline": # if the input algorithm is 'BaselineOnly'
        Algo = Baseline(train)
    elif algo=="SVD": # if the input algorithm is 'SVD'
        Algo = SVD(n_epochs=20, n_factors=50, random_state=42)

    Algo.fit(train)

    cross_validate(Algo, data_folds, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # get non-rated movie list and top_n movies of expected rating to recommend
    unseen_lst = get_unseen(df_rating, df_credit, userId)
    top_movies_preds = recomm_movie(df_credit, Algo, userId, unseen_lst, top_n=10)

    print("\n===for user {}===".format(userId))
    print("<Top 10 Recommended Movies>")
    i = 1
    for top_movie in top_movies_preds:
        print(i)
        print("title: ", top_movie[2])
        print("estimated rating: ", top_movie[1])
        print()
        i += 1

# Test
# algorithm: BaselineOnly, userId: 9
user_based(credit, rating, 'baseline', 9)
print("==================================================\n")
# algorithm: SVD, userId: 3
user_based(credit, rating, 'SVD', 3)