# %%
import tensorflow.compat.v1 as tf

# %%
tf.disable_eager_execution()

# %%
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

hidden_nodes = 5

w1 = tf.Variable(tf.random_normal([2, hidden_nodes]))
b1 = tf.Variable(tf.random_normal([hidden_nodes]))
z1 = tf.add(tf.matmul(x, w1), b1)
a1 = tf.sigmoid(z1)

w2 = tf.Variable(tf.random_normal([hidden_nodes, 1]))
z2 = tf.matmul(a1, w2)
a2 = tf.sigmoid(z2)

cross_entropy = tf.square(a2 - y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# %%
dataset = {
    "x": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    "y": [[0.0], [1.0], [1.0], [0.0]],
}

# %%
init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

# %%
for epoch in range(1, 2001):
    v = session.run([optimizer, loss], feed_dict={x: dataset["x"], y: dataset["y"]})

    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, Loss: {v[1]}")

# %%
for i in range(4):
    y_pred = session.run(a2, feed_dict={x: [dataset["x"][i]]})

    print(f"Input: {dataset['x'][i]}, Output: {y_pred}")

# %%
