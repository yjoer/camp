# %%
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# %%
dataset = fetch_20newsgroups(random_state=26)

# %%
text = dataset["data"][19]

# %% [markdown]
# ## Bag-of-Words

# %%
vectorizer = CountVectorizer()
vectorizer.fit([text])

print(vectorizer.vocabulary_)

# %%
vector = vectorizer.transform([text])

print(type(vector))
print(vector.shape)
print(vector.toarray())

# %% [markdown]
# ## N-Grams

# %%
unigrams = nltk.word_tokenize(text)
bigrams = list(nltk.ngrams(unigrams, 2))

print(len(unigrams))
print(len(bigrams))
print(bigrams)

# %%
