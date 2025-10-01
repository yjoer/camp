# %%
import io
import os
import pickle
import random
import re

import fsspec
import numpy as np
import pandas as pd
import spacy
from minio import Minio
from nltk.lm import Laplace
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import everygrams
from tqdm.auto import tqdm

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

# %%
minio = Minio(
    os.environ.get("S3_ENDPOINT", "").split("//")[-1],
    access_key=os.environ.get("S3_ACCESS_KEY_ID"),
    secret_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    secure=False,
)

# %%
dataset_url = "s3://datasets/gutenberg/20250803"
protocol = fsspec.utils.get_protocol(dataset_url)
fs = fsspec.filesystem(protocol, **storage_options)

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Sampling

# %%
url_100 = "s3://datasets/gutenberg/100.parquet"
url_100_exists = fs.exists(url_100)

# %%
if not url_100_exists:
    objs = minio.list_objects(
        bucket_name="datasets",
        prefix="gutenberg/20250810.zip/",
        recursive=True,
        extra_headers={"x-minio-extract": "true"},
    )

    ids = []
    for obj in objs:
        if m := re.search(r"pg(\d+)", obj.object_name):
            ids.append(int(m.group(1)))

# %%
if not url_100_exists:
    df = pd.read_csv(
        filepath_or_buffer="s3://datasets/gutenberg/pg_catalog.csv",
        storage_options=storage_options,
    )

    df_en = df[df["Text#"].isin(ids)]
    df_en = df_en[df_en["Language"] == "en"]

# %%
if not url_100_exists:
    rng = np.random.default_rng(26)
    files = rng.choice(df_en["Text#"], size=100)
    files = [
        minio.get_object(
            bucket_name="datasets",
            object_name=f"gutenberg/20250810.zip/{f}/pg{f}.txt",
            request_headers={"x-minio-extract": "true"},
        )
        for f in files
    ]
    texts = [f.read() for f in files]

# %%
if not url_100_exists:
    df = pd.DataFrame.from_dict({"texts": texts})
    df.to_parquet(
        path=url_100,
        engine="pyarrow",
        index=False,
        storage_options=storage_options,
    )

# %% [markdown]
# ### Sentence Segmentation

# %%
url_100_sentences = "s3://datasets/gutenberg/100_sentences.parquet"
url_100_sentences_exists = fs.exists(url_100_sentences)

# %%
if not url_100_sentences_exists:
    df = pd.read_parquet(path=url_100, storage_options=storage_options)

# %%
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
nlp.max_length = 5000000
print(nlp.pipe_names)

# %%
if not url_100_sentences_exists:
    texts = []
    sentences = []

    for t in tqdm(df.itertuples()):
        text = t.texts.decode("utf-8-sig")
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        texts.append(text)

    with nlp.select_pipes(enable=["sentencizer"]):
        docs = nlp.pipe(texts, n_process=-1, batch_size=8)
        for doc in tqdm(docs, total=len(texts)):
            sentences.extend([sent.text for sent in doc.sents])

# %%
if not url_100_sentences_exists:
    df = pd.DataFrame.from_dict({"sentences": sentences})
    df.to_parquet(
        path=url_100_sentences,
        engine="pyarrow",
        index=False,
        storage_options=storage_options,
    )

# %% [markdown]
# ### Word Tokenization

# %%
url_100_sentence_tokens = "s3://datasets/gutenberg/100_sentence_tokens.parquet"
url_100_sentence_tokens_exists = fs.exists(url_100_sentence_tokens)

# %%
if not url_100_sentence_tokens_exists:
    df = pd.read_parquet(url_100_sentences, storage_options=storage_options)

# %%
if not url_100_sentence_tokens_exists:
    sentences = df.sentences.to_list()
    sentence_tokens = []

    with nlp.select_pipes(enable=[]):
        docs = nlp.pipe(sentences, n_process=-1, batch_size=8)

        for doc in tqdm(docs, total=len(sentences)):
            tokens = [token.text for token in doc]
            if tokens[-1] == ".":
                tokens.pop()
            sentence_tokens.append(tokens)

# %%
if not url_100_sentence_tokens_exists:
    df = pd.DataFrame.from_dict({"sentence_tokens": sentence_tokens})
    df.to_parquet(
        path=url_100_sentence_tokens,
        engine="pyarrow",
        index=False,
        storage_options=storage_options,
    )

# %% [markdown]
# ## Language Modeling

# %%
lm_url = "s3://datasets/gutenberg/lm_laplace.pkl"
lm_url_exists = fs.exists(lm_url)

# %%
if not lm_url_exists:
    df = pd.read_parquet(url_100_sentence_tokens, storage_options=storage_options)
    sentence_tokens = df.sentence_tokens.to_list()
    print(sum(len(t) for t in sentence_tokens))

# %%
list(pad_both_ends(["this", "is", "a", "sentence"], n=2))

# %%
list(everygrams(pad_both_ends(["this", "is", "a", "sentence"], n=2), max_len=3))

# %%
if not lm_url_exists:
    train, vocab = padded_everygram_pipeline(5, sentence_tokens)

# %%
if not lm_url_exists:
    lm = Laplace(5)
    lm.fit(train, vocab)

# %%
if not lm_url_exists:
    memory_buffer = io.BytesIO()
    pickle.dump(lm, memory_buffer)
    memory_buffer.seek(0)
    object_name = "gutenberg/lm_laplace.pkl"

    minio.put_object(
        bucket_name="datasets",
        object_name=object_name,
        data=memory_buffer,
        length=memory_buffer.getbuffer().nbytes,
        content_type="application/octet-stream",
    )

# %%
response = minio.get_object(
    bucket_name="datasets",
    object_name="gutenberg/lm_laplace.pkl",
)

memory_buffer = io.BytesIO(response.read())
lm = pickle.load(memory_buffer)  # noqa: S301
del memory_buffer


# %%
def generate(text_seed: list[str], random_seed: int) -> str:
    sentence = text_seed[:]
    random_seed = random.Random(random_seed)  # noqa: S311

    while True:
        token = lm.generate(1, text_seed=sentence, random_seed=random_seed)
        if token == "</s>":  # noqa: S105
            sentence.append(".")
            break

        sentence.append(token)

    sentence = " ".join(sentence)
    return re.sub(r"\s+([,.'])", r"\1", sentence)


# %%
generate(text_seed=["I", "have", "a"], random_seed=26)

# %%
generate(text_seed=["This", "is", "a"], random_seed=26)

# %%
generate(text_seed=["On", "a", "starry", "night"], random_seed=26)

# %%
