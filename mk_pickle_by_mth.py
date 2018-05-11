import os
import pickle
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
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


            for i in range(len(lines)):
                season=sorted_lines[i:i+3]
                mths=[]
                vmins=[]
                vmaxs=[]


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


def mk_img(datalistfolder, savedir, min_max_dir):
    #savedir
    savedir=os.path.join(savedir,'pickle_by_mth')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)


    #dictionary by height or temperature
    dic_min_maxv_by_hp=mk_dic_min_maxv(min_max_dir)
    mthlists=os.listdir(datalistfolder)
    mthlists=sorted(mthlists)
    
    for mthlist in mthlists:
        mthlistfile=os.path.join(datalistfolder, mthlist)
        mth=os.path.splitext(mthlist)[0]
        mthdir=os.path.join(savedir, mth)

        pmth=int(mth)-1
        amth=int(mth)+1
        if pmth == 0:
            pmth+=12
        if amth > 12:
            amth=amth%12

        model='{}_{}_{}'.format('{0:0=2d}'.format(pmth), mth, '{0:0=2d}'.format(amth))

        if not os.path.isdir(mthdir):
            os.makedirs(mthdir)

        with open(mthlistfile, 'r') as f:
            datapaths=f.readlines()

        for i in range(int(len(datapaths)/80)):
            datapathsets=datapaths[80*i:80*(i+1)]
            date=datapathsets[0].strip('\n').split('.')[1][:8]
            datepickle=np.zeros((80,224*224*3))
            datepicklepath=os.path.join(mthdir, '{}.pickle'.format(date))

            for j, datapath in enumerate(datapathsets):
                datapath=datapath.strip('\n')
                namess=datapath.split('/')
                htfolder=namess[-4]
                binfilename=os.path.splitext(namess[-1])[0]

                vmin=dic_min_maxv_by_hp[htfolder][model]['min']
                vmax=dic_min_maxv_by_hp[htfolder][model]['max']



                with open(datapath, 'rb') as f:
                    rawarr=np.fromfile(f, dtype=np.float32, count=-1)
                    rawarr=rawarr.reshape(95,95)
                    rawarr=np.flip(rawarr, 0)

                    imgpath=os.path.join(savedir, binfilename+'.png')
                    plt.imshow(rawarr, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
                    plt.savefig(imgpath)
                    plt.clf()

                    img=Image.open(imgpath).convert('RGB')
                    img_size=img.size
                    box=(144,59,img_size[0]-143,img_size[1]-58)
                    img_crop=img.crop(box)
                    img_resize=img_crop.resize((224,224))
                    imgarr=np.asarray(img_resize)
                    imgarr1d=imgarr.reshape(224*224*3)
                    datepickle[j]=imgarr1d
                    os.remove(imgpath)
                    print('##{} : {} is appended to {}.pickle'.format(j, binfilename, date))

            
            with open(datepicklepath, 'wb') as f:
                pickle.dump(datepickle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('{} : pickle {} set is generated'.format(i, date))        



if __name__=='__main__':
    mk_img('/media/fdalab/TOSHIBA EXT2/weather_analysis/raw_list_by_mth', '/media/fdalab/TOSHIBA EXT2/weather_analysis', '/media/fdalab/TOSHIBA EXT2/weather_analysis/DABA')