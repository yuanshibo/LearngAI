import  tensorflow as tf
import numpy as np
from matplotlib import cm

# hello = tf.constant('hello,tensorflow')
# sess = tf.Session()
# print(sess.run(hello))
# a = tf.constant(10)
# b = tf.constant(32)
# print(sess.run(a+b))

# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.],[2.]])
# product = tf.matmul(matrix1, matrix2)
# print(sess.run(product))
# sess.close()
# from __future__ import print_function
# import tensorflow as tf

# # Set up a linear classifier.
# classifier = tf.estimator.LinearClassifier()

# # Train the model on some example data.
# classifier.train(input_fn=train_input_fn, steps=2000)

# # Use it to predict.
# predictions = classifier.predict(input_fn=predict_input_fn)

# x1 = np.linspace(-1,1,num = 10)
# x2 = np.linspace(1,10,num = 10,retstep = True)
# print(x1)
# print(x2)
# print("length of x1 is %d" % len(x1))
# print("length of x2 is %d" % len(x2))

colors = [cm.coolwarm(x) for x in np.linspace(1, 10, 10)] 
print(colors)

print('111111')