# Basic Linear regression example
# Wrote by MagmaTart (Soomin Lee)
# Last modified : 2017.11.28

import tensorflow as tf

tf.set_random_seed(9297)

dataX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataY = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

# Our Hypothesis
hypothesis = X * W + b

# mean-square error cost function
cost = tf.reduce_mean(tf.square(Y - hypothesis))

# Use Gradient Descent Optimizer
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3000):
       # Train one step
        _, c = sess.run([trainer, cost], feed_dict={X:dataX, Y:dataY})
        
        if step % 200 == 0:
            print(c)

    # Test
    print(sess.run(hypothesis, feed_dict={X:[100]}))

