'''Introductory Module for Understanding DataTypes'''

import tensorflow as tf
sess = tf.Session()
print("\n\nOutput: \n")
print("#########################\n")
###################
#   Code Here     #
###################

cons = tf.constant(3.0,dtype = tf.float32)
plac = tf.placeholder(dtype = tf.float32)
var = tf.Variable(5.0,dtype =tf.float32)
init = tf.global_variables_initializer()
sess.run(init)
print("cons= "+str(cons)+"\tplac= "+str(plac)+"\tvar= "+str(var) )
print("\ncons= "+ str(sess.run(cons))  +"\tplac= "+ str(sess.run(plac,{plac : 4.0})) +"\tvar= "+ str(sess.run(var)))

print("\n#########################\n")
