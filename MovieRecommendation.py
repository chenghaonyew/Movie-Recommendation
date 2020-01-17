import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return data[data.index == index]["title"].values[0]

def get_index_from_title(title):
	return data[data.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File with pandas extension
data = pd.read_csv("movie_dataset.csv")


##Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']


##Step 3: Create a column in data which combines all selected features
# get rid of 'NaN' columns
# fillna() - fill all NaN with thing within bracket
for feature in features:
	data[feature] = data[feature].fillna('')

# combine all the features into 1 string to make 'cosine_similarity' works
def combine_features(row):
	# try except to see the issue
	try:
		return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
	except:
		print ("Error"), row

# apply function to all the row of the data, axis=1 pass each row individually
data["combined_features"] = data.apply(combine_features, axis = 1)


##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(data["combined_features"])


##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"


## Step 6: Get index of this movie from its title
# get index of the selected movie in list
movie_index = get_index_from_title(movie_user_likes)

# find all similar movies
similar_movies = list(enumerate(cosine_sim[movie_index]))


## Step 7: Get a list of similar movies in descending order of similarity score
# sort the list based on the similarity_score
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse = True)


## Step 8: Print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
	print (get_title_from_index(movie[0]))
	i=i+1
	if i>50:
		break
