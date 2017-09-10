'''Training using optmizers in train API of TF'''

#Import Packages
import tensorflow as tf
import numpy as np

#Create Session
sess = tf.Session()

print("\n\nOutput: \n")
print("#########################\n")
###################
#   Code Here     #
###################

#Initialize Model Variables
W = tf.Variable([0.5],tf.float32)
b = tf.Variable(0.0,tf.float32)
num_iter = 100

#Initialize Model I/P and O/P
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#Compute prediction and loss
pred = tf.add(tf.multiply(W,X),b)
loss = tf.reduce_sum(tf.square(pred-y))

#Update Parameters Using optmizers
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

#Model Training Data
X_train = [1,2,3,4]
y_train = [2,3,4,5]

#Start Training
init = tf.global_variables_initializer()
sess.run(init)

for i in range(num_iter):
    sess.run(train,feed_dict = {X:X_train,y:y_train})

W_updated, b_updated, loss_updated = sess.run([W,b,loss],feed_dict= {X:X_train,y:y_train})

print("W:\t%s\tb:\t%s\tLoss:\t%s"%(W_updated, b_updated,loss_updated))

print("\n#########################\n")
