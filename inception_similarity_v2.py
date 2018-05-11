import tensorflow as tf 
import os
import pickle
from datetime import datetime as ddt 
import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from inceptionv1 import *
from pprint import pprint


def mk_dic_min_maxv(min_max_dir):
    if not min_max_dir.endswith('/'):
        min_max_dir+='/'
    
    dic_min_maxv_by_hp={}
    for file in os.listdir(min_max_dir):
        file_r=file.upper()
        if file.endswith('_src'):
            file_r=file.replace('_src','')
            os.rename(file, file_r)
                  
        with open(min_max_dir+file, 'r') as f:
            dic_min_maxv_by_ss={} 
            lines=f.readlines()
            
            sorted_lines=[lines[i] for i in range(len(lines))]+lines[:2]
            # pprint(sorted_lines)

            for i in range(len(lines)):
            	season=sorted_lines[i:i+3]
            	mths=[]
            	vmins=[]
            	vmaxs=[]
            	# print(season)

            	for mth_min_max in season:
            		mth_min_max=mth_min_max.split(' ')
            		mths.append(mth_min_max[0])
            		vmins.append(mth_min_max[1])
            		vmaxs.append(mth_min_max[2])

            	vmin=min(vmins)
            	vmax=max(vmaxs)
            	mths='{}_{}_{}'.format(mths[0], mths[1], mths[2])
            	dic_min_maxv_by_ss[mths]={'min':int(vmin),'max':int(vmax)}

        dic_min_maxv_by_hp[file_r]=dic_min_maxv_by_ss

    return dic_min_maxv_by_hp

def feature_extract(input_date, data_root_dir, pre_prams_dir):

	#0. define variables for use
	""" 
	min_max_dir : directory of files which contain minimum and maximum values of data
	rawdir : directory of raw data
	dictlyrpath : directory of pretrained_parameter npy file
	rank : the number of rank to be shown as result
	model : one of the 12 models corresponding to input date
	len_input : size of input data in inception v-1 model
	feature : directory of extracted features 
	features_model_pkl : directory of pickle file of corresponding model which contains all features of previous days of input_date
	dates_model_pkl : Similar with features_model_pkl, but it is saved with dates information

	"""
	min_max_dir='/media/fdalab/TOSHIBA EXT1/KU_Project/KU_Weather/DABA'
	utcs=['00', '06', '12', '18']
	heights=['H100','H200','H300','H400','H500','H600','H700','H800','H900','H1000']
	temps=['T100','T200','T300','T400','T500','T600','T700','T800','T900','T1000'] 
	featuredir='/media/fdalab/TOSHIBA EXT2/weather_analysis/feature'
	rank=20
	len_input=224*224*3

	#next_date
	
	inputdate=ddt.strptime(str(input_date), '%Y%m%d')
	deltatime=datetime.timedelta(days=1)
	tomorrow=inputdate+deltatime
	tomorrow=tomorrow.strftime('%Y%m%d')

	yr=int(str(input_date)[:4])
	mth=int(str(input_date)[4:6])
	dt=int(str(input_date)[6:])

	next_yr=int(tomorrow[:4])
	next_mth=int(tomorrow[4:6])
	next_dt=int(tomorrow[6:])

	pmth=mth-1
	if pmth == 0:
		pmth+=12

	amth=mth+1
	if amth > 12 :
		amth=amth%12

	model='{}_{}_{}'.format('{0:0=2d}'.format(pmth), mth, '{0:0=2d}'.format(amth))


    
	# features_model_path=os.path.join(featuredir, model)
 	# dates_model_path=os.path.join(featuredir, '{}dates'.format(model))

	features_model_pkl=os.path.join(featuredir, '{}_demo.pickle'.format(model))
	dates_model_pkl=os.path.join(featuredir, '{}dates_demo.pickle'.format(model))
	dic_min_maxv=mk_dic_min_maxv(min_max_dir)

	#1. change raw data into image pickle
	imgpickles=[]
	next_imgpickles=[]
	for utc in utcs:
		print(utc)
		erautc='ERA_INT_KRU_{}'.format(utc)
		rawdirpath=os.path.join(rawdir,erautc)
		for height in heights:
			print(height)

			#get min, max value of each height
			hmin=dic_min_maxv[height][model]['min']
			hmax=dic_min_maxv[height][model]['max']

			#get image pickle of inputdate of each height
			h_rawdirpath=os.path.join(rawdirpath, height, str(yr), str(mth))
			h_bin='era_easia_{}_anal.{}{}_s2.bin'.format(height.lower(), input_date, utc)
			hbinpath=os.path.join(h_rawdirpath, h_bin)
			
			with open(hbinpath, 'rb') as f:
				rawarr=np.fromfile(f, dtype=np.float32, count=-1)
				rawarr=rawarr.reshape(95,95)
				rawarr=np.flip(rawarr, 0)


			imgpath=os.path.join(rawdir, os.path.splitext(h_bin)[0]+'.png')
			plt.imshow(rawarr, cmap='nipy_spectral', vmin=hmin, vmax=hmax)
			plt.savefig(imgpath)
			plt.clf()

			img=Image.open(imgpath).convert('RGB')
			img_size=img.size
			box=(144,59,img_size[0]-143,img_size[1]-58)
			img_crop=img.crop(box)
			img_resize=img_crop.resize((224,224))
			imgarr=np.asarray(img_resize)
			imgarr1d=imgarr.reshape(224*224*3)

			imgpickles.append(imgarr1d)
			os.remove(imgpath)

			# get image pickle of tomorrow of each height
			h_rawdirpath=os.path.join(rawdirpath, height, str(next_yr), str(next_mth))
			h_bin='era_easia_{}_anal.{}{}_s2.bin'.format(height.lower(), tomorrow, utc)
			hbinpath=os.path.join(h_rawdirpath, h_bin)

			with open(hbinpath, 'rb') as f:
				rawarr=np.fromfile(f, dtype=np.float32, count=-1)
				rawarr=rawarr.reshape(95,95)
				rawarr=np.flip(rawarr, 0)

			imgpath=os.path.join(rawdir, os.path.splitext(h_bin)[0]+'.png')
			plt.imshow(rawarr, cmap='nipy_spectral', vmin=hmin, vmax=hmax)
			plt.savefig(imgpath)
			plt.clf()

			img=Image.open(imgpath).convert('RGB')
			img_size=img.size
			img_crop=img.crop(box)
			img_resize=img_crop.resize((224,224))
			imgarr=np.asarray(img_resize)
			imgarr1d=imgarr.reshape(224*224*3)

			next_imgpickles.append(imgarr1d)
			os.remove(imgpath)

		for temp in temps:
			print(temp)
			tmin=dic_min_maxv[temp][model]['min']
			tmax=dic_min_maxv[temp][model]['max']

			#get image pickle of inputdate of each temperature 
			t_rawdirpath=os.path.join(rawdirpath, temp, str(yr), str(mth))
			t_bin='era_easia_{}_anal.{}{}_s2.bin'.format(temp.lower(), input_date, utc)		
			tbinpath=os.path.join(t_rawdirpath, t_bin)

			with open(tbinpath, 'rb') as f:
				rawarr=np.fromfile(f, dtype=np.float32, count=-1)
				rawarr=rawarr.reshape(95,95)
				rawarr=np.flip(rawarr, 0)

			imgpath=os.path.join(rawdir, os.path.splitext(t_bin)[0]+'.png')
			plt.imshow(rawarr, cmap='nipy_spectral', vmin=tmin, vmax=tmax)
			plt.savefig(imgpath)
			plt.clf()

			img=Image.open(imgpath).convert('RGB')
			img_size=img.size
			box=(144,59,img_size[0]-143,img_size[1]-58)
			img_crop=img.crop(box)
			img_resize=img_crop.resize((224,224))
			imgarr=np.asarray(img_resize)
			imgarr1d=imgarr.reshape(224*224*3)

			imgpickles.append(imgarr1d)
			os.remove(imgpath)

			#get image pickle of tomorrow date of each temperature
			t_rawdirpath=os.path.join(rawdirpath, temp, str(next_yr), str(next_mth))
			t_bin='era_easia_{}_anal.{}{}_s2.bin'.format(temp.lower(), tomorrow, utc)		
			tbinpath=os.path.join(t_rawdirpath, t_bin)

			with open(tbinpath, 'rb') as f:
				rawarr=np.fromfile(f, dtype=np.float32, count=-1)
				rawarr=rawarr.reshape(95,95)
				rawarr=np.flip(rawarr, 0)

			imgpath=os.path.join(rawdir, os.path.splitext(t_bin)[0]+'.png')
			plt.imshow(rawarr, cmap='nipy_spectral', vmin=tmin, vmax=tmax)
			plt.savefig(imgpath)
			plt.clf()

			img=Image.open(imgpath).convert('RGB')
			img_size=img.size
			box=(144,59,img_size[0]-143,img_size[1]-58)
			img_crop=img.crop(box)
			img_resize=img_crop.resize((224,224))
			imgarr=np.asarray(img_resize)
			imgarr1d=imgarr.reshape(224*224*3)

			next_imgpickles.append(imgarr1d)
			os.remove(imgpath)

	#2. feature extraction
	"""Implementing inception-v1 model and pre-trained parameter, features of input-dates are extracted"""
	dictlyr=np.load(dictlyrpath, encoding='latin1').item()
	params_pre=reformat_params(dictlyr)
	X=tf.placeholder(tf.float32, [None, len_input])
	feature=arxt(X, params_pre)

	##loading the existing features
	with open(features_model_pkl, 'rb') as f:
		features_model=pickle.load(f)
		features_model=np.asarray(features_model)

	with open(dates_model_pkl, 'rb') as f:
		dates_model=pickle.load(f)
		dates_model=np.asarray(dates_model)

	existing_feature_vectors=[features_model, dates_model]
	
	"""extracting features from image pickles of input date"""
	with tf.Session() as sess:
		init=tf.global_variables_initializer()
		sess.run(init)

		for i in range(len(imgpickles)):
			print(i)
			imgpkl=imgpickles[i]
			imgpkl=imgpkl.reshape(-1, len(imgpkl))
			feature_img=sess.run(feature, feed_dict={X:imgpkl})

			if i == 0:
				features=feature_img
			else:
				features=np.concatenate((features, feature_img), axis=0)

	"""extracting features from image pickles of tomorrow"""
	with tf.Session() as sess2:
		init=tf.global_variables_initializer()
		sess2.run(init)

		for i in range(len(next_imgpickles)):
			print(i)
			nextimgpkl=next_imgpickles[i]
			nextimgpkl=nextimgpkl.reshape(-1, len(nextimgpkl))
			next_feature_img=sess2.run(feature, feed_dict={X:nextimgpkl})

			if i == 0:
				next_features=next_feature_img
			else:
				next_features=np.concatenate((next_features, next_feature_img), axis=0)


	input_feature_vector=[features, str(input_date)]
	next_feature_vector=[features, str(tomorrow)]

	feature_vector=[input_feature_vector, next_feature_vector]

	return feature_vector, existing_feature_vectors

def compare_similarity(feature_vector, existing_feature_vectors):

	rank=20
	ftr, input_date=feature_vector[0]
	ftr=ftr.reshape(1,-1)
	nextftr, tomorrow=feature_vector[1]
	nextftr=nextftr.reshape(1,-1)

	ftrs, dates=existing_feature_vectors


	###one_day
	MSE=np.mean((ftrs-ftr)**2, axis=1)
	MSE_index=np.argsort(MSE)[:rank]
	MSE_rank=dates[MSE_index]
	MSE_vals=MSE[MSE_index]
	one=list(zip(MSE_rank, MSE_vals))


	###split HT
	ftrs_H=np.concatenate([ftrs[:,:10*1024], ftrs[:,20*1024:30*1024], ftrs[:,40*1024:50*1024], ftrs[:,60*1024:70*1024]], axis=1)
	ftrs_T=np.concatenate([ftrs[:,10*1024:20*1024], ftrs[:,30*1024:40*1024], ftrs[:,50*1024:60*1024], ftrs[:,70*1024:80*1024]], axis=1)

	ftr_H=np.concatenate([ftr[:,:10*1024], ftr[:,20*1024:30*1024], ftr[:,40*1024:50*1024], ftr[:,60*1024:70*1024]], axis=1)
	ftr_T=np.concatenate([ftr[:,10*1024:20*1024], ftr[:,30*1024:40*1024], ftr[:,50*1024:60*1024], ftr[:,70*1024:80*1024]], axis=1)

	MSE_H=np.mean((ftrs_H-ftr_H)**2, axis=1)
	MSE_index_H=np.argsort(MSE_H)[:rank]
	MSE_rank_H=dates[MSE_index_H]
	MSE_val_H=MSE_H[MSE_index_H]
	one_H=list(zip(MSE_rank_H, MSE_val_H))


	MSE_T=np.mean((ftrs_T-ftr_T)**2, axis=1)
	MSE_index_T=np.argsort(MSE_T)[:rank]
	MSE_rank_T=dates[MSE_index_T]
	MSE_val_T=MSE_T[MSE_index_T]
	one_T=list(zip(MSE_rank_T, MSE_val_T))

	###two_day
	sorted_ind=np.argsort(dates)
	sorted_dates=dates[sorted_ind]
	sorted_ftrs=ftrs[sorted_ind]

	ftr2=np.concatenate((ftr, nextftr), axis=1)
	ftrs2=np.concatenate((sorted_ftrs[:-1], sorted_ftrs[1:]), axis=1)

	#MSE
	MSE2=np.mean((ftrs2-ftr2)**2, axis=1)
	MSE2_index=np.argsort(MSE2)[:rank]
	MSE2_rank=sorted_dates[MSE2_index]
	MSE2_val=MSE2[MSE2_index]
	two=list(zip(MSE2_rank, MSE2_val))
	
	#split HT twoday
	ftrs2_H=np.concatenate([ftrs2[:,:10*2048], ftrs2[:,20*2048:30*2048], ftrs2[:,40*2048:50*2048], ftrs2[:,60*2048:70*2048]], axis=1)
	ftrs2_T=np.concatenate([ftrs2[:,10*2048:20*2048], ftrs2[:,30*2048:40*2048], ftrs2[:,50*2048:60*2048], ftrs2[:,70*2048:80*2048]], axis=1)

	ftr2_H=np.concatenate([ftr2[:,:10*2048], ftr2[:,20*2048:30*2048], ftr2[:,40*2048:50*2048], ftr2[:,60*2048:70*2048]], axis=1)
	ftr2_T=np.concatenate([ftr2[:,10*2048:20*2048], ftr[:,30*2048:40*2048], ftr2[:,50*2048:60*2048], ftr2[:,70*2048:80*2048]], axis=1)

	MSE2_H=np.mean((ftrs2_H-ftr2_H)**2, axis=1)
	MSE2_index_H=np.argsort(MSE2_H)[:rank]
	MSE2_rank_H=sorted_dates[MSE2_index_H]
	MSE2_val_H=MSE_H[MSE2_index_H]
	two_H=list(zip(MSE2_rank_H, MSE2_val_H))


	MSE2_T=np.mean((ftrs2_T-ftr2_T)**2, axis=1)
	MSE2_index_T=np.argsort(MSE2_T)[:rank]
	MSE2_rank_T=sorted_dates[MSE2_index_T]
	MSE2_val_T=MSE2_T[MSE2_index_T]
	two_T=list(zip(MSE2_rank_T, MSE2_val_T))

	return one, one_H, one_T, two, two_H, two_T




if __name__ == '__main__':
	rawdir='/media/fdalab/TOSHIBA EXT2/weather_analysis'
	dictlyrpath='/home/fdalab/Desktop/KU_Project/KU_Weather/Data_2017/2nd_work/codes/googlenet.npy'
	feature_vector, existing_feature_vectors=feature_extract(20161031, rawdir, dictlyrpath)
	compare_similarity(feature_vector, existing_feature_vectors)
