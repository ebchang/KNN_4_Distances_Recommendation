# -*- coding: utf-8 -*-
"""
@author: evanchang
"""

import re
#import metrics
import math
import time
import pickle
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

# Get's users in the object of .data
class User:
    def __init__(self, id, age, sex, occupation, zip_code):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip_code = zip_code
        self.avg_r = 0.0

# Makes each of the 1 of the 19 genres an object of .data
class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
    unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
    drama, fantasy, film_noir, horror, musical, mystery ,romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)

# Gets Ratings from .data set
class Rating:
    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.timestamp= timestamp

# The dataset class helps you to load files and create User, Item and Rating objects
class Dataset:
    
        
    def get_users(self, file, u):
        f = open(file, "r")
        input_file = f.read()
        entries = re.split("\n+", input_file)
        for entry in entries:
            emit_users = entry.split('|', 5)
            if len(emit_users) == 5:
                u.append(User(emit_users[0], emit_users[1], emit_users[2],\
                emit_users[3], emit_users[4]))
        return u
        f.close()

    def get_items(self, file, i):
        f = open(file, encoding='utf-8', errors='ignore')
        input_file = f.read()
        entries = re.split("\n+", input_file)
        for entry in entries:
            emit_items = entry.split('|', 24)
            if len(emit_items) == 24:
                i.append(Item(emit_items[0], emit_items[1], emit_items[2], emit_items[3],\
                emit_items[4], emit_items[5], emit_items[6], emit_items[7], \
                emit_items[8], emit_items[9], emit_items[10], emit_items[11], \
                emit_items[12], emit_items[13], emit_items[14], emit_items[15],\
                emit_items[16], emit_items[17],emit_items[18], emit_items[19],\
                emit_items[20], emit_items[21], emit_items[22], emit_items[23]))
        f.close()

    def get_ratings(self, file, r):
        f = open(file, "r")
        input_file = f.read()
        entries = re.split("\n+", input_file)
        for entry in entries:
            emit_ratings = entry.split('\t', 4)
            if len(emit_ratings) == 4:
                r.append(Rating(emit_ratings[0], emit_ratings[1],\
                emit_ratings[2], emit_ratings[3]))
        f.close()

#Everything is organized from here. ===========================================
 
# Store data in arrays
user_object_list = []
item_object_list = []
rating_object_list = []
rating_test_object_list = []

# Getting the specifics for what is going to be used in the metric algorithms
d = Dataset()
d.get_users("u.user", user_object_list)
d.get_items("u.item", item_object_list)
d.get_ratings("u.data", rating_object_list)
d.get_ratings("u1.test", rating_test_object_list)
 
n_users = len(user_object_list)
n_items = len(item_object_list)

#DATA SET is now made with users, items, and ratings separated and stored as objects
#===================================================================================
 
# The utility matrix stores the rating for each user-item pair in the
 # matrix form.
 #Makes a matrix of the lens of users X items.
matrix_users_items = np.zeros((n_users, n_items))
for r in rating_object_list:
    matrix_users_items[r.user_id - 1][r.item_id - 1] = r.rating

print (matrix_users_items)

test_matrix = np.zeros((n_users, n_items))
for r in rating_test_object_list:
    test_matrix[r.user_id - 1][r.item_id - 1] = r.rating

# Perform clustering on items by genre
movie_genre = []
for movie in item_object_list:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

#Movie Genre is an array of the type of movies for each item. stored in
#an array list of lists matrix
movie_genre = np.array(movie_genre)
cluster = KMeans(n_clusters=2)
cluster.fit_predict(movie_genre)

#This will take all of the genre ratings and average them per user.
#Will come out with the average rating per user and store them in an array 
#utility_clustered

utility_clustered_manhattan = []


for i in range(0, n_users):
    average = np.zeros(19)
    tmp = []
    for m in range(0, 19):
        tmp.append([])
    for j in range(0, n_items):
        if matrix_users_items[i][j] != 0:
            tmp[cluster.labels_[j] - 1].append(matrix_users_items[i][j])
    for m in range(0, 19):
        if len(tmp[m]) != 0:
            average[m] = np.mean(tmp[m])
        else:
            average[m] = 0
    utility_clustered_manhattan.append(average)
    
utility_clustered_manhattan = np.array(utility_clustered_manhattan)

for i in range(0, n_users):
    x = utility_clustered_manhattan[i]
    user_object_list[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    

#Find the Manhattan distance between users. 
#Find the Manhattan distance
#==============================================================================
manhattan_matrix = np.zeros((n_users, n_users))
 
def manhattan(x,y):

    den1 =0
    den2 = 0
    A = utility_clustered_manhattan[x-1]
    B = utility_clustered_manhattan[y-1]
    den1 = sum((a) for a in A if a>0)
    den2 = sum((b)  for b in B if b>0)
    den = abs(den1 - den2)
    if den ==0:
        return 0
    else:
        return den

for i in range(0, n_users):
    for j in range(0, n_users):
        if i !=j:
            manhattan_matrix[i][j] = manhattan(i+1, j+1)
            sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, manhattan_matrix[i][j]))
            sys.stdout.flush()
            time.sleep(0.0005)
            
print(sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, manhattan_matrix[i][j])))

print(manhattan_matrix)
def similarity_score_manhattan(user1, user2):
    both_viewed = []
    
    for item in manhattan_matrix[user1]:
        if item in manhattan_matrix[user2]:
            both_viewed[item] = 1 
    
    if len(both_viewed) == 0:
        return 0
        
    sum_of_manhattan_distance = []
    for item in manhattan_matrix[user1]:
        if item in manhattan_matrix[user2]:
            sum_of_manhattan_distance.append(pow(manhattan_matrix[user1][item]))
            sum_of_manhattan_distance = sum(sum_of_manhattan_distance)
        return 1/(1+math.sqrt(sum_of_manhattan_distance))
        

for i in range(0, n_users):
    for j in range(0, n_users):
        if i !=j:
            manhattan_matrix[i][j] = manhattan(i+1, j+1)
            sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1,manhattan_matrix[i][j]))
            sys.stdout.flush()
            time.sleep(0.00005)
print("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1,manhattan_matrix[i][j]))

print(manhattan_matrix)
 # Guesses the ratings that user with id, user_id, might give to item with id, i_id.
 # We will consider the top_n similar users to do this.
def norm_manhattan():
     normalize = np.zeros((n_users, 19))
     for i in range(0, n_users):
        for j in range(0, 19):
            if utility_clustered_manhattan[i][j] != 0:
                normalize[i][j] = utility_clustered_manhattan[i][j] - user_object_list[i].avg_r
            else:
                normalize[i][j] = float('Inf')
                return normalize

def guess_manhattan(user_id, i_id, top_n):
    similarity_manhattan = []
    for i in range(0, n_users):
        if i+1 != user_id:
            similarity_manhattan.append(manhattan_matrix[user_id-1][i])
    temp = norm_manhattan()
    temp = np.delete(temp, user_id-1, 0)
    top = [x for (y,x) in sorted(zip(similarity_manhattan,temp), key=lambda pair: pair[0], reverse=True)]
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id-1] != float('Inf'):
             s += top[i][i_id-1]
             c += 1
    genre = user_object_list[user_id-1].avg_r if c == 0 else s/float(c) + user_object_list[user_id-1].avg_r
    if genre < 1.0:
        return 1.0
    elif genre > 5.0:
        return 5.0
    else:
        return genre
 
utility_copy = np.copy(utility_clustered_manhattan)
for i in range(0, n_users):
    for j in range(0, 19):
        if utility_copy[i][j] == 0:
            sys.stdout.write("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
            sys.stdout.flush()
            time.sleep(0.00005)
            utility_copy[i][j] = guess_manhattan(i+1, j+1, 150)
print ("\rGuessing [User:Rating] = [%d:%d]" % (i, j))

print (utility_copy)

pickle.dump( utility_copy, open("utility_matrix_manhattan.pkl", "wb"))

# Predict ratings for u.test and find the mean absolute error
true = []
pred = []
f = open('test_manhattan_2kK.txt', 'w')
for i in range(0, n_users):
    for j in range(0, n_items):
        if test_matrix[i][j] > 0:
            f.write("%d, %d, %.4f\n" % (i+1, j+1, utility_copy[i][cluster.labels_[j]-1]))
            true.append(test_matrix[i][j])
            pred.append(utility_copy[i][cluster.labels_[j]-1])
f.close()

print ("Mean absolute Error: %f" % mean_absolute_error(true, pred))


