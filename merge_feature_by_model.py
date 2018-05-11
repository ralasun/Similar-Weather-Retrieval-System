import os
import pickle
import numpy as np 

def merge_feature(featuredir, savedir):
	savedir=os.path.join(savedir, 'feature_by_model')
	if not os.path.isdir(savedir):
		os.mkdir(savedir)

	#mthfeaturelists : 1월부터 12월까지의 12개의 월이 적혀 있는 리스트 
	mthfeaturelists=os.listdir(featuredir)
	mthfeaturelists=sorted(mthfeaturelists)

	for mth in mthfeaturelists:
		if mth == '02':
			pmth=int(mth)-1
			amth=int(mth)+1

			if pmth==0:
				pmth+=12
			if amth>12:
				amth=amth%12

			model='{}_{}_{}'.format('{0:0=2d}'.format(pmth), mth, '{0:0=2d}'.format(amth))

			pmthdir=os.path.join(featuredir, "{0:0=2d}".format(pmth))
			mthdir=os.path.join(featuredir, mth)
			amthdir=os.path.join(featuredir, '{0:0=2d}'.format(amth))
			

			##merge features by model
			modelfeatures=[]
			modeldates=[]

			#pmth : 모델의 첫번째 달에 대해 feature와 날짜정보를 각각 리스트에 첨가. 
			pmthftrlists=os.listdir(pmthdir)
			pmthftrlists=sorted(pmthftrlists)
			for i in range(len(pmthftrlists)):
				dateftr=pmthftrlists[i]
				date=os.path.splitext(dateftr)[0]
				dateftrpath=os.path.join(pmthdir, dateftr)

				with open(dateftrpath, 'rb') as f:
					datefeature=pickle.load(f)

				modelfeatures.append(datefeature)
				modeldates.append(date)

			#mth : 모델의 두번째 달에 대해 feature와 날짜정보를 각각 리스트에 첨가
			mthftrlists=os.listdir(mthdir)
			mthftrlists=sorted(mthftrlists)
			for i in range(len(mthftrlists)):
				dateftr=mthftrlists[i]
				date=os.path.splitext(dateftr)[0]
				dateftrpath=os.path.join(mthdir, dateftr)

				with open(dateftrpath, 'rb') as f:
					datefeature=pickle.load(f)

				modelfeatures.append(datefeature)
				modeldates.append(date)

			#amth : 모델의 세번째 달에 대해 feature와 날짜정보를 각각 리스트에 첨가 
			amthftrlists=os.listdir(amthdir)
			amthftrlists=sorted(amthftrlists)
			for i in range(len(amthftrlists)):
				dateftr=amthftrlists[i]
				date=os.path.splitext(dateftr)[0]
				dateftrpath=os.path.join(amthdir, dateftr)

				with open(dateftrpath, 'rb') as f:
					datefeature=pickle.load(f)

				modelfeatures.append(datefeature)
				modeldates.append(date)

			modelfeaturespath=os.path.join(savedir, '{}.pickle'.format(model))
			modeldatespath=os.path.join(savedir, '{}dates.pickle'.format(model))

			with open(modelfeaturespath, 'wb') as f:
				pickle.dump(modelfeatures, f)
			with open(modeldatespath, 'wb') as f:
				pickle.dump(modeldates, f)

			print('{}model feature is merged'.format(model))



if __name__=='__main__':
	merge_feature('/media/fdalab/TOSHIBA EXT2/KU_Project/KU_Weather/3rd_work/01_02_03/feature3', '/media/fdalab/TOSHIBA EXT2/KU_Project/KU_Weather/3rd_work/01_02_03')