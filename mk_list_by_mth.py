#import libraries
import os, sys
import pickle
from pprint import pprint


#this function make buch of files which contain all dates of binary files (19790101~20170430)
def make_date_list(raw_data_folder, startyr, endyr):
	#raw_data_folder : 일기도 데이터가 들어 있는 사용자 지정 폴더 경로 
	#startyr : 일기도 데이터에서 첫 년도 
	#endyr : 일기도 데이터에서 마지막 년도 

	if not raw_data_folder.endswith('/'):
		raw_data_folder+='/'

	d_list=[]
	date_list=[]
	mth_list=["01","02","03","04","05","06","07","08","09","10","11","12"]
	for i in range(startyr, endyr):
		for mth in mth_list:
			 #path : 00UTC의 h100폴더에 들어있는 모든 경로를 가지고와서, 일기도 데이터에 들어있는 모든 날짜를 리스트에 첨가 
			path=raw_data_folder+'ERA_INT_KRU_00/'+"H100/"+str(i)+'/'+mth    
			d_list+=os.listdir(path)

	for d in d_list:
		dd=d.split("_anal.")[1]
		ddd=dd.split("_s2")[0]
		date_list.append(ddd[0:8])
           
	return date_list

def make_list_by_mth(raw_data_folder, savedir, startyr, endyr):
	
	savedir=os.path.join(savedir, 'raw_list_by_mth')
	if not os.path.isdir(savedir):
		os.mkdir(savedir)

	utcs=['00', '06', '12', '18']
	heitdir=['H100','H200','H300','H400','H500','H600','H700','H800','H900','H1000']
	tempdir=['T100','T200','T300','T400','T500','T600','T700','T800','T900','T1000']
	date_list=make_date_list(raw_data_folder, startyr, endyr)
	date_list=sorted(date_list)

	for date in date_list:
 		yr=date[:4]
 		mth=date[4:6]
 		monthly_file_list=mth+'.txt'
 		savedirbymth=os.path.join(savedir, monthly_file_list)

 		for utc in utcs:
 			utcfolder=os.path.join(raw_data_folder, 'ERA_INT_KRU_{}'.format(utc))
 			for height in heitdir:
 				utc_ht_fld=os.path.join(utcfolder, height, yr, mth)
 				h_binfile='era_easia_{}_anal.{}{}_s2.bin'.format(height.lower(), date, utc)
 				h_binfiledir=os.path.join(utc_ht_fld, h_binfile)

 				with open(savedirbymth, 'a') as mfl:
 					mfl.write(h_binfiledir+'\n')
 					print('%s is written in text files' %(h_binfiledir))

 			for temp in tempdir:
 				utc_tmp_fld=os.path.join(utcfolder, temp, yr, mth)
 				t_binfile='era_easia_{}_anal.{}{}_s2.bin'.format(temp.lower(), date, utc)
 				t_binfiledir=os.path.join(utc_tmp_fld, t_binfile)

 				with open(savedirbymth, 'a') as mfl:
 					mfl.write(t_binfiledir+'\n')
 					print('%s is written in text files' %(t_binfiledir))

 				


if __name__=='__main__':
 	make_list_by_mth('/media/fdalab/TOSHIBA EXT2/weather_analysis', '/media/fdalab/TOSHIBA EXT2/weather_analysis',2010, 2017)




