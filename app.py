import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle as pk
import recommendation_functions as rf


# Create the `requirements.txt` file:

requirements= '\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None))
with open('/Users/jlma/Documents/Curso de Data Science/08-Recommender Systems/movies_recommendation_system/requirements.txt', 'w') as output:
    output.write(requirements)

# Loading the DATA:
path= "/Users/jlma/Documents/Curso de Data Science/08-Recommender Systems/movies_recommendation_system/data/"

links_df    = pd.read_csv(f'{path}links.csv')
movies_df   = pd.read_csv(f'{path}movies.csv')
ratings_df  = pd.read_csv(f'{path}ratings.csv')
tags_df     = pd.read_csv(f'{path}tags.csv')

# variables declaration:

minimun_rating = 3 
minimun_reviews = 200 
top_number_of_movies_recommended = 37

Correlation_Method = "pearson" #@param ["pearson", "kendall", "spearman"]
movieId = 2 

user_id = 120

# Model calculation

rf.recommendation_popularity(ratings_df,movies_df,minimun_rating,minimun_reviews,top_number_of_movies_recommended)

rf.recommendation_byMoviesCorrelation(ratings_df, movies_df, movieId, Correlation_Method, minimun_reviews, minimun_rating, top_number_of_movies_recommended)

rf.recommendation_byUsersSimilarity (ratings_df, movies_df, user_id,top_number_of_movies_recommended)