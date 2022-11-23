#%%
import string
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import tabulate
from scipy import stats
from wordcloud import WordCloud 
from typing import Dict, Text
from ast import literal_eval
from datetime import datetime
from collections import Counter
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape

import warnings
warnings.filterwarnings('ignore')

#%%
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
movies = pd.read_csv('movies_metadata.csv').\
                     drop(['belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'status', 'title', 'video'], axis=1).\
                     drop([19730, 29503, 35587]) # Incorrect data type

movies['id'] = movies['id'].astype('int64')

df = movies.merge(keywords, on='id').\
    merge(credits, on='id')

df['original_language'] = df['original_language'].fillna('')
df['runtime'] = df['runtime'].fillna(0)
df['tagline'] = df['tagline'].fillna('')

df.dropna(inplace=True)

def get_text(text, obj='name'):
    text = literal_eval(text)
    
    if len(text) == 1:
        for i in text:
            return i[obj]
    else:
        s = []
        for i in text:
            s.append(i[obj])
        return ', '.join(s)

#%%
df['genres'] = df['genres'].apply(get_text)
df['production_companies'] = df['production_companies'].apply(get_text)
df['production_countries'] = df['production_countries'].apply(get_text)
df['crew'] = df['crew'].apply(get_text)
df['spoken_languages'] = df['spoken_languages'].apply(get_text)
df['keywords'] = df['keywords'].apply(get_text)

# New columns
df['characters'] = df['cast'].apply(get_text, obj='character')
df['actors'] = df['cast'].apply(get_text)

df.drop('cast', axis=1, inplace=True)
df = df[~df['original_title'].duplicated()]
df = df.reset_index(drop=True)
# %%
df.head()

#%%
df.head(1)
#%%
df.info()
# %%
df['release_date'] = pd.to_datetime(df['release_date'])
df['budget'] = df['budget'].astype('float64')
df['popularity'] = df['popularity'].astype('float64')


# %%
sns.displot(data=df, x='release_date', kind='hist', kde=True, line_kws={'lw': 3}, aspect=3)
plt.title('Total Released Movie by Date', fontsize=18, weight=600)

#%%
spoken_languages_list = []
for i in df['spoken_languages']:
    if i != '':
        spoken_languages_list.extend(i.split(', '))

country_list = []
for i in df['production_countries']:
    if i != '':
        country_list.extend(i.split(', '))

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 10))

# Spoken language plot
df_plot1 = pd.DataFrame(Counter(spoken_languages_list).most_common(10), columns=['language', 'total']).sort_values(by='total', ascending=True)
ax1.hlines(y=df_plot1['language'], xmin=0, xmax=df_plot1['total'], alpha=0.7, linewidth=2)
ax1.scatter(x=df_plot1['total'], y=df_plot1['language'], s = 75)
ax1.set_title('\nTop 10 Spoken Languages\nin Movies\n', fontsize=15, weight=600)
for i, value in enumerate(df_plot1['total']):
    ax1.text(value+1000, i, value, va='center', fontsize=10, weight=600)

# Country plot
df_plot6 = pd.DataFrame(Counter(country_list).most_common(10), columns=['name', 'total']).sort_values(by='total', ascending=True)
ax2.hlines(y=df_plot6['name'], xmin=0, xmax=df_plot6['total'], alpha=0.7, linewidth=2)
ax2.scatter(x=df_plot6['total'], y=df_plot6['name'], s = 75)
ax2.set_title('\nTop 10 Production Countries\n', fontsize=15, weight=600)
for i, value in enumerate(df_plot6['total']):
    ax2.text(value+900, i, value, va='center', fontsize=10, weight=600)
    
sns.despine()
plt.tight_layout()
# %%
sns.relplot(data=df, x='vote_average', y='popularity', size='vote_count',
            sizes=(20, 200), aspect=2)
plt.title('The Relationship Between Rating and Popularity', fontsize=25, weight=600)

# %%
genres_list = []
for i in df['genres']:
    genres_list.extend(i.split(', '))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

df_plot = pd.DataFrame(Counter(genres_list).most_common(5), columns=['genre', 'total'])
ax = sns.barplot(data=df_plot, x='genre', y='total', ax=axes[0])
ax.set_title('Top 5 Genres in Movies', fontsize=18, weight=600)
sns.despine()

df_plot_full = pd.DataFrame([Counter(genres_list)]).transpose().sort_values(by=0, ascending=False)
df_plot.loc[len(df_plot)] = {'genre': 'Others', 'total':df_plot_full[6:].sum()[0]}
plt.title('Percentage Ratio of Movie Genres', fontsize=18, weight=600)
wedges, texts, autotexts = axes[1].pie(x=df_plot['total'], labels=df_plot['genre'], autopct='%.2f%%',
                                       textprops=dict(fontsize=14), explode=[0,0,0,0,0,0.1])

for autotext in autotexts:
    autotext.set_weight('bold')

axes[1].axis('off')

#%%
df_plot = pd.DataFrame(Counter(genres_list).most_common(5), columns=['genre', 'total'])
df_plot = df[df['genres'].isin(df_plot['genre'].to_numpy())]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6))

plt.suptitle('Data Distribution Across Top 5 Genres', fontsize=18, weight=600)
for i, y in enumerate(['runtime', 'popularity', 'budget', 'revenue']):
    sns.stripplot(data=df_plot, x='genres', y=y, ax=axes.flatten()[i])

plt.tight_layout()
# %%
