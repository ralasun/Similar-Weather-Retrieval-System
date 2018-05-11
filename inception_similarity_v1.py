import os
import pickle as pickle
import numpy as np 
from pprint import pprint


def compare_similarity(date_to_be_compared, featuredir):

	rank=20
	date_to_be_compared=str(date_to_be_compared)
	mth=date_to_be_compared[4:6]


	#해당 하는 모델의 feature경로 찾기
	pmth=int(mth)-1
	amth=int(mth)+1
	if pmth == 0:
		pmth+=12
	if amth > 12:
		amth=amth%12
	model='{}_{}_{}'.format('{0:0=2d}'.format(pmth), mth, '{0:0=2d}'.format(amth))
	modelfeaturedir=os.path.join(featuredir, '{}.pickle'.format(model))
	modeldatesdir=os.path.join(featuredir, '{}dates.pickle'.format(model))

	with open(modelfeaturedir, 'rb') as f:
		ftrs=pickle.load(f)

	with open(modeldatesdir, 'rb') as f:
		dates=pickle.load(f)

	ftrs=np.asarray(ftrs)
	dates=np.asarray(dates)


	###one_day
	indice=np.where(dates==date_to_be_compared)
	ftr=ftrs[indice[0][0]]
	ftr=ftr.reshape(1,-1)
	MSE=np.mean((ftrs-ftr)**2, axis=1)
	MSE_index=np.argsort(MSE)[1:rank+1]
	MSE_rank=dates[MSE_index]
	MSE_vals=MSE[MSE_index]
	one=list(zip(MSE_rank, MSE_vals))


	###split HT
	ftrs_H=np.concatenate([ftrs[:,:10*1024], ftrs[:,20*1024:30*1024], ftrs[:,40*1024:50*1024], ftrs[:,60*1024:70*1024]], axis=1)
	ftrs_T=np.concatenate([ftrs[:,10*1024:20*1024], ftrs[:,30*1024:40*1024], ftrs[:,50*1024:60*1024], ftrs[:,70*1024:80*1024]], axis=1)

	ftr_H=np.concatenate([ftr[:,:10*1024], ftr[:,20*1024:30*1024], ftr[:,40*1024:50*1024], ftr[:,60*1024:70*1024]], axis=1)
	ftr_T=np.concatenate([ftr[:,10*1024:20*1024], ftr[:,30*1024:40*1024], ftr[:,50*1024:60*1024], ftr[:,70*1024:80*1024]], axis=1)

	indice_h=np.where(dates==date_to_be_compared)
	ftr_H=ftrs_H[indice_h[0][0]]
	MSE_H=np.mean((ftrs_H-ftr_H)**2, axis=1)
	MSE_index_H=np.argsort(MSE_H)[1:rank+1]
	MSE_rank_H=dates[MSE_index_H]
	MSE_val_H=MSE_H[MSE_index_H]
	one_H=list(zip(MSE_rank_H, MSE_val_H))

	indice_t=np.where(dates==date_to_be_compared)
	ftr_T=ftrs_T[indice_t[0][0]]
	MSE_T=np.mean((ftrs_T-ftr_T)**2, axis=1)
	MSE_index_T=np.argsort(MSE_T)[1:rank+1]
	MSE_rank_T=dates[MSE_index_T]
	MSE_val_T=MSE_T[MSE_index_T]
	one_T=list(zip(MSE_rank_T, MSE_val_T))

	###two_day
	sorted_ind=np.argsort(dates)
	sorted_dates=dates[sorted_ind]
	sorted_ftrs=ftrs[sorted_ind]

	indice2=np.where(sorted_dates==date_to_be_compared)
	ftrs2=np.concatenate((sorted_ftrs[:-1], sorted_ftrs[1:]), axis=1)
	ftr2=ftrs2[indice2[0][0]]
	ftr2=ftr2.reshape(1,-1)

	#MSE
	MSE2=np.mean((ftrs2-ftr2)**2, axis=1)
	MSE2_index=np.argsort(MSE2)[1:rank+1]
	MSE2_rank=sorted_dates[MSE2_index]
	MSE2_val=MSE2[MSE2_index]
	two=list(zip(MSE2_rank, MSE2_val))
	
	#split HT twoday
	ftrs2_H=np.concatenate([ftrs2[:,:10*2048], ftrs2[:,20*2048:30*2048], ftrs2[:,40*2048:50*2048], ftrs2[:,60*2048:70*2048]], axis=1)
	ftrs2_T=np.concatenate([ftrs2[:,10*2048:20*2048], ftrs2[:,30*2048:40*2048], ftrs2[:,50*2048:60*2048], ftrs2[:,70*2048:80*2048]], axis=1)

	ftr2_H=np.concatenate([ftr2[:,:10*2048], ftr2[:,20*2048:30*2048], ftr2[:,40*2048:50*2048], ftr2[:,60*2048:70*2048]], axis=1)
	ftr2_T=np.concatenate([ftr2[:,10*2048:20*2048], ftr[:,30*2048:40*2048], ftr2[:,50*2048:60*2048], ftr2[:,70*2048:80*2048]], axis=1)


	MSE2_H=np.mean((ftrs2_H-ftr2_H)**2, axis=1)
	MSE2_index_H=np.argsort(MSE2_H)[1:rank+1]
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
	featuredir='C:\\Users\\user\\Desktop\\drive-download-20180109T111545Z-001\\feature_by_model'
	one, oneh, onet,two, twoh, twot=compare_similarity(20161031, featuredir)
