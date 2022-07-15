
## 1) Popularity/Quality based recommmender system:


def recommendation_popularity(ratings_df,movies_df,minimun_rating,minimun_reviews,top_number_of_movies_recommended):
  import pandas as pd  
  rating_movieId_mean = pd.DataFrame(ratings_df.groupby('movieId')['rating'].mean()).sort_values("rating", ascending=False)
  rating_movieId_mean = rating_movieId_mean[rating_movieId_mean['rating'] >= minimun_rating]

  rating_movieId_mean['reviews'] = ratings_df.groupby('movieId')['rating'].count()
  reviews= rating_movieId_mean[rating_movieId_mean['reviews'] >= minimun_reviews].sort_values("reviews", ascending=False)

  top= pd.DataFrame(reviews).head(top_number_of_movies_recommended)

  top_movies_names_df= pd.merge(top, movies_df, on='movieId', how='inner')
    
    # Show recommendations

  print(f"Our TOP {top_number_of_movies_recommended} movies: ")

  recommended_movies = pd.DataFrame(top_movies_names_df['title'])
  recommended_movies.columns = ['Recommended_Movies']
  recommended_movies.set_index('Recommended_Movies', inplace=True)
  return recommended_movies

## 2) Method: **`Correlation`**: using MOVIES correlation


def recommendation_byMoviesCorrelation(ratings_df, movies_df, movieId, correlation_method, minimun_reviews, minimun_rating, top_number_of_movies_recommended):
  import pandas as pd
  movies_ratings_df = ratings_df.loc[ratings_df['rating'] >= minimun_rating]
  movies_crosstab = pd.pivot_table(data=movies_ratings_df, values='rating', index='userId', columns='movieId')
  cross_ratings = movies_crosstab[movieId]

  similar_ratings = movies_crosstab.corrwith(cross_ratings,method=correlation_method)

  corr_df = pd.DataFrame(similar_ratings, columns=[f'Correlation_{correlation_method}'])
  corr_df.dropna(inplace=True)

  rating = pd.DataFrame(movies_ratings_df.groupby('movieId')['rating'].mean())
  rating['reviews'] = movies_ratings_df.groupby('movieId')['rating'].count()


  movies_corr_summary = corr_df.join(rating['reviews'])
  movies_corr_summary.drop(movieId, inplace=True) # drop the inputed movie itself

  top = movies_corr_summary[movies_corr_summary['reviews']>=minimun_reviews].sort_values(f'Correlation_{correlation_method}', ascending=False).head(top_number_of_movies_recommended)

  top_movies_names_df= pd.merge(top, movies_df, on='movieId', how='inner')

  selected_movie_name = movies_df[movies_df['movieId'] == movieId]['title']


  print(f"""Based on your recent selection:

  {selected_movie_name.values[0]}

  this are the movies that we would like to recommend:""")

  recommended_movies = pd.DataFrame(top_movies_names_df['title'])
  recommended_movies.columns = ['Recommended_Movies']
  recommended_movies.set_index('Recommended_Movies', inplace=True)
  return recommended_movies

## 3) Method: **`Cosine Similarities`**: using USERS similarity


def recommendation_byUsersSimilarity (ratings_df, movies_df, user_id,top_number_of_movies_recommended):
  import pandas as pd
  movies =  movies_df[['movieId', 'title']]

  # Create a Pivot DataFrame, where the values are the rating values and the rows= userId  & columns= movieId

  users_items = pd.pivot_table(data=ratings_df, 
                                  values='rating', 
                                  index='userId', 
                                  columns='movieId')

  # Replace NaNs with zeros, The cosine similarity can't be computed with NaN's

  users_items.fillna(0, inplace=True)
  users_items

  # Import and Apply the cosine similarities

  from sklearn.metrics.pairwise import cosine_similarity

  user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                  columns=users_items.index, 
                                  index=users_items.index)

  weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))

  #  Find the movies user selected has not rated. We will exclude our user, since we don't want to include them on the weights.


  not_watched_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
  weighted_averages = pd.DataFrame(not_watched_movies.T.dot(weights), columns=["predicted_rating"])
  weighted_averages

  # Show recommendations

  print(f"Specially for YOU: ")

  recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
  top_movies_names_df = recommendations.sort_values("predicted_rating", ascending=False).head(top_number_of_movies_recommended)


  recommended_movies = pd.DataFrame(top_movies_names_df['title'])
  recommended_movies.columns = ['Recommended_Movies']
  recommended_movies.set_index('Recommended_Movies', inplace=True)
  return recommended_movies