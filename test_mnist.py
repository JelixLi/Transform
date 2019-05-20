import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import psutil 
import objgraph
from tensorflow.python.framework import ops


def bin_activ(x):
	bin_x = tf.reshape(tf.sign(x),[tf.shape(x)[0],tf.reduce_prod(tf.shape(x)[1:])])
	alpha = tf.reduce_mean(tf.reshape(tf.abs(x),[tf.shape(x)[0],tf.reduce_prod(tf.shape(x)[1:])]),axis=1)
	alpha = tf.tile(tf.reshape(alpha,[tf.shape(x)[0],1]),[1,tf.reduce_prod(tf.shape(x)[1:])])
	return tf.reshape(bin_x * alpha,tf.shape(x))

def bin_activ_input(x):
	return bin_activ(x)

def bin_activ_weight(x):
	return tf.reshape(bin_activ(tf.reshape(x,[tf.shape(x)[3],tf.reduce_prod(tf.shape(x)[:3])])),tf.shape(x))

def bin_conv(x,w):
	bin_x = bin_activ(x)
	bin_w = bin_activ(w)

def logic_multiply(x,y):
	lx = tf.cast(x,tf.int32)
	ly = tf.cast(y,tf.int32)
	return tf.cast(tf.multiply(lx,ly),tf.bool)

def MN_activ(x):
	mn_x = tf.reshape(x,[tf.shape(x)[0],tf.reduce_prod(tf.shape(x)[1:])])
	mean = tf.tile(tf.reshape(tf.reduce_mean(mn_x,axis=1),[tf.shape(x)[0],1]),[1,tf.reduce_prod(tf.shape(x)[1:])])
	delta = tf.tile(tf.reshape(tf.sqrt(tf.reduce_mean(tf.square(mn_x - mean),axis=1)),[tf.shape(x)[0],1]),[1,tf.reduce_prod(tf.shape(x)[1:])])

	sumA = tf.reduce_sum(tf.where(tf.less_equal(mn_x,-delta), mn_x, tf.zeros_like(mn_x)),axis=1)
	sumB = tf.reduce_sum(tf.where(logic_multiply(tf.greater(mn_x,-delta),tf.less(mn_x,0)) , mn_x, tf.zeros_like(mn_x)),axis=1)
	sumC = tf.reduce_sum(tf.where(logic_multiply(tf.greater_equal(mn_x,0),tf.less(mn_x,delta)) , mn_x, tf.zeros_like(mn_x)),axis=1)
	sumD = tf.reduce_sum(tf.where(tf.greater_equal(mn_x,delta), mn_x, tf.zeros_like(mn_x)),axis=1)

	numA = tf.count_nonzero(tf.less_equal(mn_x,-delta),axis=1)
	numB = numA + tf.count_nonzero(logic_multiply(tf.greater(mn_x,-delta),tf.less(mn_x,0)),axis=1)
	numC = numB + tf.count_nonzero(logic_multiply(tf.greater_equal(mn_x,0),tf.less(mn_x,delta)),axis=1)
	numD = tf.tile(tf.reshape(tf.reduce_prod(tf.shape(x)[1:]),[1]),[tf.shape(x)[0]])

	numA = tf.cast(numA,tf.float32)
	numB = tf.cast(numB,tf.float32)
	numC = tf.cast(numC,tf.float32)
	numD = tf.cast(numD,tf.float32)

	num_n = tf.divide((sumC-sumB),(numC - numA))
	num_m = tf.divide((sumD - sumA),(numA + numD -numC))

	num_n = tf.tile(tf.reshape(num_n,[tf.shape(x)[0],1]),[1,tf.reduce_prod(tf.shape(x)[1:])])
	num_m = tf.tile(tf.reshape(num_m,[tf.shape(x)[0],1]),[1,tf.reduce_prod(tf.shape(x)[1:])])

	re_mn_x = mn_x
	re_mn_x = tf.where(tf.less_equal(mn_x,-delta), -num_m,re_mn_x)
	re_mn_x = tf.where(logic_multiply(tf.greater(mn_x,-delta),tf.less(mn_x,0)), -num_n,re_mn_x)
	re_mn_x = tf.where(logic_multiply(tf.greater_equal(mn_x,0),tf.less(mn_x,delta)), num_n,re_mn_x)
	re_mn_x = tf.where(tf.greater_equal(mn_x,delta), num_m,re_mn_x)

	return tf.reshape(re_mn_x,tf.shape(x))


def MN_activ_input(x):
	return MN_activ(x)

def MN_activ_weight(x):
	return tf.reshape(MN_activ(tf.reshape(x,[tf.shape(x)[3],tf.reduce_prod(tf.shape(x)[:3])])),tf.shape(x))

bin_w_module = tf.load_op_library('./trans_weight.so')
bin_m_module = tf.load_op_library('./trans_input.so')

# bin_w_module = tf.load_op_library('./trans_gpu/trans_weight_gpu.so')
# bin_m_module = tf.load_op_library('./trans_gpu/trans_input_gpu.so')

bin_w_tri_module = tf.load_op_library('./trans_weight_tri.so')
bin_m_tri_module = tf.load_op_library('./trans_input_tri.so')

# bin_w_bin_module = tf.load_op_library('./trans_weight_bin.so')
# bin_m_bin_module = tf.load_op_library('./trans_input_bin.so')

bin_m_bin_module = tf.load_op_library('./trans_gpu/trans_input_gpu_bin.so')
bin_w_bin_module = tf.load_op_library('./trans_gpu/trans_weight_gpu_bin.so')

@ops.RegisterGradient("BinActivW")
def _bin_activ_w_grad(op, grad):
    return grad

@ops.RegisterGradient("BinActivM")
def _bin_activ_m_grad(op, grad):
    return grad

@ops.RegisterGradient("BinActivWTR")
def _bin_activ_w_grad(op, grad):
    return grad

@ops.RegisterGradient("BinActivMTR")
def _bin_activ_m_grad(op, grad):
    return grad

@ops.RegisterGradient("BinActivWBI")
def _bin_activ_w_grad(op, grad):
    return grad

@ops.RegisterGradient("BinActivMBI")
def _bin_activ_m_grad(op, grad):
    return grad




def memory_monitor():
	print psutil.virtual_memory()
	print psutil.swap_memory()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape,name):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial,name=name)

def bias_varriable(shape,name):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial,name=name)



def conv2d(x,W):
	# new_x = tf.nn.batch_normalization(x,mean=0,variance=0.1,offset=0,scale=1,variance_epsilon=1e-5)
	# new_x = bin_m_module.bin_activ_m(tf.nn.batch_normalization(x,mean=0,variance=0.1,offset=0,scale=1,variance_epsilon=1e-5))
	# new_w = bin_w_module.bin_activ_w(W)
	# return tf.nn.conv2d(new_x,new_w,strides=[1,1,1,1],padding='SAME')

	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv2d(x,W,training):
  	new_x = bin_m_module.bin_activ_m(tf.layers.batch_normalization(x, training=training))
  	new_w = bin_w_module.bin_activ_w(W)
	# new_x = bin_m_tri_module.bin_activ_mtr(tf.layers.batch_normalization(x, training=training))
	# new_w = bin_w_tri_module.bin_activ_wtr(W)
	# new_x = bin_m_bin_module.bin_activ_mbi(tf.layers.batch_normalization(x, training=training))
	# new_w = bin_w_bin_module.bin_activ_wbi(W)
	# new_x = tf.layers.batch_normalization(x, training=training)
	# new_x = MN_activ_input(new_x)
	# new_w = MN_activ_weight(W)
	return tf.nn.conv2d(new_x,new_w,strides=[1,1,1,1],padding='SAME')




def max_pool_2X2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



x = tf.placeholder("float",[None,784])
x_image = tf.reshape(x,[-1,28,28,1])

y_ = tf.placeholder("float",[None,10])

training = tf.placeholder(tf.bool)

W_conv1 = weight_variable([5,5,1,32],"W_conv1")
b_conv1 = bias_varriable([32],"b_conv1")

W_conv2 = weight_variable([5,5,32,64],"W_conv2")
b_conv2 = bias_varriable([64],"b_conv2")

W_fc1 = weight_variable([7*7*64,1024],"W_fc1")
b_fc1 = bias_varriable([1024],"b_fc1")

W_fc2 = weight_variable([1024,10],"W_fc2")
b_fc2 = bias_varriable([10],"b_fc2")


h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1,training) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,training) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
	#train_step = tf.train.MomentumOptimizer(1e-4,0.9,use_nesterov=True).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

tf.summary.scalar("loss" , cross_entropy)
summary_op = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('/home/jielix/code/py/output_graph/',sess.graph)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)



for i in range(20000):
	batch = mnist.train.next_batch(50)

	if (i+1) % 100 == 0 :
		batch_test = mnist.test.next_batch(50)
		testing_feed_dict = {x:batch_test[0],y_:batch_test[1],keep_prob:1.0,training:False}
		print "step: %d training accurancy: %g" % (i+1,accuracy.eval(feed_dict=testing_feed_dict))
        # summary_writer.add_summary(sess.run(summary_op,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}), i)
	training_feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5,training:True}
	train_step.run(feed_dict=training_feed_dict)



save_path = saver.save(sess, "/home/jielix/code/py/test_model.ckpt")

sess.close()