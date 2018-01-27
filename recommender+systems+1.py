
# coding: utf-8

# **  A basic recommendation system by suggesting items that are most similar to a particular item, in this case, movies. This is not a true robust recommendation system, to describe it more accurately,it just tells you what movies/items are most similar to your movie choice. **

# ### Import the libraries

# In[52]:


import numpy as np
import pandas as pd


# ### Get the Data 

# In[53]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep= '\t', names= column_names)


# In[54]:


df.head()


# In[55]:


movie_titles = pd.read_csv('Movie_Id_Titles')
movie_titles.head()


# In[56]:


df = pd.merge(df, movie_titles, on= 'item_id')


# In[57]:


df.head()


# ### Import visualization libraries 

# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[59]:


sns.set_style('white')


# In[60]:


df.groupby('title')['rating'].mean().sort_values(ascending = False)


# In[61]:


df.groupby('title')['rating'].count().sort_values(ascending = False).head()


# In[62]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[63]:


ratings.head()


# In[64]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[65]:


ratings.head()


# In[66]:


plt.figure(figsize=(12,4))
ratings['num of ratings'].hist(bins= 70)
plt.xlabel('Number Of Ratings')
plt.ylabel('Number Of Movies')


# In[67]:


plt.figure(figsize=(12,4))
ratings['rating'].hist(bins= 70)
plt.xlabel('Ratings')
plt.ylabel('Number Of Movies')


# In[68]:


sns.jointplot('rating', 'num of ratings', data= ratings, alpha= 0.5)


# ### Recommending Similar Movies 

# Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[69]:


moviemat = df.pivot_table(index= 'user_id', columns= 'title', values= 'rating')
moviemat.head()


# Most Rated Movie

# In[70]:


ratings.sort_values('num of ratings', ascending= False).head(10)


# Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

# In[71]:


ratings.head()


# Now let's grab the user ratings for those two movies:

# In[72]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())
print('\n')
print(liarliar_user_ratings.head())


# We can then use corrwith() method to get correlations between two pandas series:

# In[73]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# Let's clean this by removing NaN values and using a DataFrame instead of a series:

# In[74]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns= ['Corelation'])
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns= ['Corelation'])


# In[75]:


corr_starwars.dropna(inplace= True)
corr_starwars.head()


# In[76]:


corr_liarliar.dropna(inplace= True)
corr_liarliar.head()


# Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie). 

# In[77]:


corr_starwars.sort_values('Corelation', ascending= False).head(10)


# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[78]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now sort the values and notice how the titles make a lot more sense:

# In[79]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Corelation', ascending= False).head(10)


# Now the same for the comedy Liar Liar:

# In[80]:


corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar.head()


# In[81]:


corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Corelation', ascending= False).head(10)

