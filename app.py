import pandas as pd
import recommendation_functions as rf

links_df    = pd.read_csv('/data/links.csv')
movies_df   = pd.read_csv('/data/movies.csv')
ratings_df  = pd.read_csv('/data/ratings.csv')
tags_df     = pd.read_csv('/data/tags.csv')

#variables:

minimun_rating = 3 
minimun_reviews = 200 
top_number_of_movies_recommended = 37

Correlation_Method = "pearson" #@param ["pearson", "kendall", "spearman"]
movieId = 2 

user_id = 120

rf.recommendation_popularity(minimun_rating,minimun_reviews,top_number_of_movies_recommended)

rf.recommendation_byMoviesCorrelation(movieId, Correlation_Method, minimun_reviews, minimun_rating, top_number_of_movies_recommended)

rf.recommendation_byUsersSimilarity (user_id,top_number_of_movies_recommended)