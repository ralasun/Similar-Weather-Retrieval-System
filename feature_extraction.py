import tensorflow as tf 
import os
import pickle
import numpy as np 
from inceptionv1 import *
from pprint import pprint

def feature_extract(pickledir, savedir, pre_params_dir):
	savedir=os.path.join(savedir, 'feature_by_mth')
	if not os.path.isdir(savedir):
		os.mkdir(savedir)

	len_input=224*224*3
	dictlyr=np.load(pre_params_dir, encoding='latin1').item()
	params_pre=reformat_params(dictlyr)

	X=tf.placeholder(tf.float32, [None, len_input])
	feature=arxt(X, params_pre)

	mthpicklelists=os.listdir(pickledir)
	mthpicklelists=sorted(mthpicklelists)

	for mth in mthpicklelists:
		mthfeaturedir=os.path.join(savedir, mth)
		mthpickledir=os.path.join(pickledir, mth)

		if not os.path.isdir(mthfeaturedir):
			os.makedirs(mthfeaturedir)

		mthpickles=os.listdir(mthpickledir)
		mthpickles=sorted(mthpickles)

		with tf.Session() as sess:
			init=tf.global_variables_initializer()
			sess.run(init)

			for i in range(len(mthpickles)):
				datepkl=mthpickles[i]
				date=os.path.splitext(datepkl)[0]
				datepklpath=os.path.join(mthpickledir, datepkl)

				with open(datepklpath, 'rb') as f:
					dateinput=pickle.load(f)


				for j in range(dateinput.shape[0]):
					eachinput=dateinput[j,:]
					eachinput=eachinput.reshape(-1, len(eachinput))
					eachfeature=sess.run(feature, feed_dict={X:eachinput})

					if j==0:
						features=eachfeature
					else:
						features=np.concatenate((features, eachfeature), axis=0)
				featurespath=os.path.join(mthfeaturedir, '{}.pickle'.format(date))
				with open(featurespath, 'wb') as f:
					pickle.dump(features, f)

				print('{} : {} feature is generated'.format(i, date))


if __name__=='__main__':
	feature_extract('/home/fdalab/Desktop/KU_Project/KU_Weather_Final/pickle_by_mth', '/home/fdalab/Desktop/KU_Project/KU_Weather_Final/','/home/fdalab/Desktop/KU_Project/KU_Weather_Final/codes_final/googlenet.npy')
