# fmt: off
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fsspec==2025.5.1",
#     "marimo",
#     "s3fs==2025.5.1",
#     "scikit-learn==1.7.0",
#     "tensorflow-cpu==2.19.0",
# ]
# ///

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import tensorflow.compat.v1 as tf
    from sklearn.preprocessing import OneHotEncoder

    from camp.datasets.mnist import FashionMNIST
    return FashionMNIST, OneHotEncoder, os, tf


@app.cell
def _(tf):
    tf.disable_eager_execution()
    return


@app.cell
def _(os):
    storage_options = {
        "endpoint_url": os.getenv("S3_ENDPOINT"),
        "key": os.getenv("S3_ACCESS_KEY_ID"),
        "secret": os.getenv("S3_SECRET_ACCESS_KEY").replace("\\", ""),
    }
    return (storage_options,)


@app.cell
def _(FashionMNIST, storage_options):
    dataset = FashionMNIST.load(
        path="s3://datasets/fashion_mnist",
        storage_options=storage_options,
        return_tensors="np",
    )

    X_train = dataset["train"] / 255
    X_test = dataset["test"] / 255
    return X_test, X_train, dataset


@app.cell
def _(OneHotEncoder, dataset):
    enc = OneHotEncoder()

    y_train = enc.fit_transform(dataset["train_labels"].reshape(-1, 1)).toarray()
    y_test = enc.transform(dataset["test_labels"].reshape(-1, 1)).toarray()
    return y_test, y_train


@app.cell
def _(tf):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    layer_1_nodes = 256
    layer_2_nodes = 256

    w1 = tf.get_variable("w1", [784, layer_1_nodes])
    b1 = tf.get_variable("b1", [layer_1_nodes], initializer=tf.zeros_initializer())
    z1 = tf.add(tf.matmul(x, w1), b1)
    a1 = tf.nn.relu(z1)

    w2 = tf.get_variable("w2", [layer_1_nodes, layer_2_nodes])
    b2 = tf.get_variable("b2", [layer_2_nodes], initializer=tf.zeros_initializer())
    z2 = tf.add(tf.matmul(a1, w2), b2)
    a2 = tf.nn.relu(z2)

    w3 = tf.get_variable("w3", [layer_2_nodes, 10])
    b3 = tf.get_variable("b3", [10], initializer=tf.zeros_initializer())
    z3 = tf.add(tf.matmul(a2, w3), b3)
    # a3 = tf.sigmoid(z3)
    a3 = z3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=a3))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return a3, loss, optimizer, x, y


@app.cell
def _(tf):
    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init)
    return (session,)


@app.cell
def _(X_train, loss, optimizer, session, x, y, y_train):
    for epoch in range(1, 11):
        v = session.run([optimizer, loss], feed_dict={x: X_train, y: y_train})

        print(f"Epoch: {epoch}, Loss: {v[1]}")
    return


@app.cell
def _(X_test, a3, session, tf, x, y, y_test):
    correct = tf.equal(tf.argmax(a3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    accuracy.eval({x: X_test, y: y_test}, session=session)
    return


if __name__ == "__main__":
    app.run()
