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

#minimun_rating = 3 
#minimun_reviews = 200 
#top_number_of_movies_recommended = 37

#correlation_method = "pearson" #@param ["pearson", "kendall", "spearman"]
#movieId = 2 

#user_id = 120

# Model calculation


#rf.recommendation_popularity(ratings_df,movies_df,minimun_rating,minimun_reviews,top_number_of_movies_recommended)

#rf.recommendation_byMoviesCorrelation(ratings_df, movies_df, movieId, correlation_method, minimun_reviews, minimun_rating, top_number_of_movies_recommended)

#rf.recommendation_byUsersSimilarity (ratings_df, movies_df, user_id,top_number_of_movies_recommended)

#Streamlit version


st.title('Movie Recommendator for Data Nerds')

st.header('Top Movies for YOU')
st.write('Movies can be sorted by minimun_rating or minimun_reviews.')


con3, con1, con2 = st.container()
with con3:
    st.write('.\n')
    top_number_of_movies_recommended = st.slider("top_number_of_movies:", 10)
    minimun_rating = st.slider("minimun_rating", value=2.5)
    minimun_reviews = st.slider("minimun_reviews", value=50)
    popularity_final_df=rf.recommendation_popularity(ratings_df,movies_df,minimun_rating,minimun_reviews,top_number_of_movies_recommended)
    st.table(data=popularity_final_df)
with con1:
    st.write('.\n')
    movieId = st.slider("movieId", value=2)
    movies_final_df = rf.recommendation_popularity(ratings_df,movies_df,minimun_rating,minimun_reviews,top_number_of_movies_recommended)
    st.table(data=movies_final_df)
with con2:
    st.write('.\n')
    user_id = st.slider("user_id", value=200)
    users_final_df = rf.recommendation_byUsersSimilarity (ratings_df, movies_df, user_id,top_number_of_movies_recommended)
    st.table(data=users_final_df)
    st.form_submit_button('Show')