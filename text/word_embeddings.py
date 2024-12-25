# %%
import gensim.downloader
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Count-Based

# %%
words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]

X = np.array(
    [
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # brown fox
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # quick brown fox
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # The quick brown fox
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # The quick brown fox jumps.
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],  # The quick brown fox jumps over.
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # jumps
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # jumps over
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # lazy dog
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # the lazy dog
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # The quick brown fox jumps over the lazy dog.
    ]
)

U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

# %%
plt.axis((-1, 1, -1, 1))

for i in range(len(words)):
    print(f"{U[i,0]:>7.4f}, {U[i, 1]:>7.4f} {words[i]}")
    plt.text(U[i, 0], U[i, 1], words[i])

# %% [markdown]
# ## Model-Based

# %% [markdown]
# ### Word2Vec

# %%
wv_path = gensim.downloader.load("word2vec-google-news-300", return_path=True)
wv = KeyedVectors.load_word2vec_format(wv_path, limit=10000, binary=True)

# %%
wv.most_similar(positive=["woman", "king"], negative=["man"], topn=5)

# %%
wv.most_similar_cosmul(positive=["woman", "king"], negative=["man"], topn=5)

# %%
wv.doesnt_match("breakfast cereal dinner lunch".split())

# %% [markdown]
# ### GloVe

# %%
gv_path = gensim.downloader.load("glove-wiki-gigaword-100", return_path=True)
gv = KeyedVectors.load_word2vec_format(gv_path, limit=10000, binary=False)

# %%
gv.most_similar(positive=["woman", "king"], negative=["man"], topn=5)

# %%
gv.most_similar_cosmul(positive=["woman", "king"], negative=["man"], topn=5)

# %%
gv.doesnt_match("breakfast cereal dinner lunch".split())

# %%
