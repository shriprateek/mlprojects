'''Implementation of simple linear regression for practicing basics'''

#Import Packages
import tensorflow as tf

#Create Session
sess = tf.Session()

print("\n\nOutput: \n")
print("#########################\n")
###################
#   Code Here     #
###################

#Initialize variables

W = tf.Variable([0.5],tf.float32)
b = tf.Variable(0.0,tf.float32)
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)

#Compute prediction and loss

pred = tf.add(tf.multiply(W,X),b)
loss = tf.reduce_sum(tf.square(pred-y))

print("Initial Loss:")
print(sess.run(loss,feed_dict = {X:[1,2,3,4],y:[2,3,4,5]}))

#Update Parameters

b_node = tf.assign(b,1)
W_node = tf.assign(W,[2])

sess.run([b_node,W_node])

print("Updated Loss:")
print(sess.run(loss,feed_dict = {X:[1,2,3,4],y:[2,3,4,5]}))


print("\n#########################\n")
