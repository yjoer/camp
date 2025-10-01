# %%
import nltk
import spacy
from IPython.display import HTML
from IPython.display import SVG
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from spacy.symbols import ORTH
from spacy.tokens import Span

# %%
# if not spacy.util.is_package("en_core_web_sm"):
#     spacy.cli.download("en_core_web_sm")

# %%
nltk.download("punkt_tab")

# %%
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
nlp.pipe_names

# %%
dataset = fetch_20newsgroups(random_state=26)

# %%
text = dataset["data"][19]
text_mini = "Apple is looking at buying U.K. startup for $1 billion"

# %% [markdown]
# ## Lexical Attributes

# %%
doc_a = nlp(text_mini)
[(token.i, token.text, token.is_alpha, token.like_num) for token in doc_a]

# %%
special_case = [{ORTH: "start"}, {ORTH: "up"}]
nlp.tokenizer.add_special_case("startup", special_case)
doc_b = nlp("U.K. startup")
[(token.i, token.text) for token in doc_b]

# %% [markdown]
# ## Sentence Segmentation

# %%
nlp_b = spacy.blank("en")
nlp_b.add_pipe("sentencizer")
nlp_b.pipe_names

# %%
doc_c = nlp_b("This is a sentence. This is another sentence.")
list(doc_c.sents)

# %% [markdown]
# ## Lemmatization

# %%
[(token.i, token.text, token.lemma_) for token in doc_a]

# %% [markdown]
# ## POS Tagging

# %%
[(token.i, token.text, token.pos_) for token in doc_a]

# %% [markdown]
# ## Dependency Parsing

# %%
SVG(spacy.displacy.render(doc_b, style="dep", jupyter=False))

# %% [markdown]
# ## Named Entity Recognition

# %%
[(token.text, token.label_) for token in doc_a.ents]

# %%
HTML(spacy.displacy.render(doc_a, style="ent", jupyter=False))

# %% [markdown]
# ## Spans

# %%
doc_d = nlp("Welcome to the Bank of China.")
doc_d.spans["sc"] = [Span(doc_d, 3, 6, "ORG"), Span(doc_d, 5, 6, "GPE")]
HTML(spacy.displacy.render(doc_d, style="span", jupyter=False))

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
unigrams = nltk.word_tokenize(text_mini)
print(len(unigrams))
print(unigrams)

# %%
bigrams = list(nltk.ngrams(unigrams, 2))
print(len(bigrams))
print(bigrams)

# %%
trigrams = list(nltk.ngrams(unigrams, 3))
print(len(trigrams))
print(trigrams)

# %%
