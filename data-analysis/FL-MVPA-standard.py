#!/usr/bin/python

import numpy as np
import sys


import glob
import time
astart=time.time()

from mvpa2.tutorial_suite import *

mask_ID=sys.argv[1]



part_list=np.loadtxt("./results/func-loc/osc-part-list.txt",dtype="str")
oscdatalist=list()
modatalist=list()

the_targets=[np.loadtxt("./results/FL-MVPA/labels/run"+str(i)+"_labels.txt",dtype="str") for i in [1,2] ]

the_sm_targets=np.hstack([np.loadtxt("./results/FL-MVPA/labels/run"+str(i)+"_sm_labels.txt",dtype="str") for i in [1,2] ])

trial_ID=np.loadtxt("./results/FL-MVPA/labels/trial_ID.txt")



n=0
for p in part_list:

        currfiles=(np.sort(glob.glob("./results/FL-MVPA/ts-inst/"+p+"/*")))
        curr_mask_name=glob.glob("./results/ROIs/"+mask_ID+".*.nii.gz")

        currmofiles=np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-movie_in_movie_run-*_tost.nii.gz"))


#       print currfiles,currmofiles,curr_mask_name


        oscdatalist.append(vstack([fmri_dataset(currfiles[i-1],mask=curr_mask_name,targets=the_targets[i-1]) for i in [1,2]]))
        oscdatalist[n].sa['chunks']=np.hstack([np.repeat(i,160) for i in [1,2]])
        oscdatalist[n].sa['participants']=np.repeat(n+1,320)
        oscdatalist[n].sa['smlabels']=the_sm_targets
        oscdatalist[n].sa['trialID']=trial_ID


        modatalist.append(vstack([fmri_dataset(currmofiles[i],mask=curr_mask_name) for i in np.arange(6)]))
        modatalist[n].sa['chunks']=np.hstack([np.repeat(i,302) for i in np.arange(1,7)])
        print time.time() - astart, p
        n=n+1

d_osc=list()
zd_osc=list()

d_movie=list()
zd_movie=list()

## Preprocessing

for p in np.arange(len(oscdatalist)):

        detrender=PolyDetrendMapper(polyord=2,chunks_attr='chunks')

        detrender.train(oscdatalist[p])
        d_osc.append(oscdatalist[p].get_mapped(detrender))

        detrender.train(modatalist[p])
        d_movie.append(modatalist[p].get_mapped(detrender))



        zscorer=ZScoreMapper(chunks_attr='chunks')

        zscorer.train(d_osc[p])
        zd_osc.append(d_osc[p].get_mapped(zscorer))

        zscorer.train(d_movie[p])
        zd_movie.append(d_movie[p].get_mapped(zscorer))





## Sample Selection

ss_osc=[ds[ds.targets!='R'] for ds in zd_osc]
sms_osc=list()

for p in np.arange(len(oscdatalist)):
        print p
        curr_avg=np.zeros((32,zd_osc[p].shape[1]))
        for t in np.unique(trial_ID)[1:]:
                print t
                curr_avg[t-1,:]=zd_osc[p].samples[np.argwhere(trial_ID==t).flatten(1),:].mean(axis=0)
        ds=ss_osc[p].copy(deep=False,sa=['targets', 'chunks','participants'],fa=['voxel_indices'])
        ds.samples=curr_avg
        sms_osc.append(ds)


n_feat=int(np.round(np.min([zd_movie[p].shape[1] for p in np.arange(len(zd_movie))])*.95))

# Rough feature selection

clf=LinearCSVMC()
cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=mean_match_accuracy)

osc_wsc_ss=[cv(ds[:,:n_feat]) for ds in ss_osc]
osc_wsc_sms=[cv(ds[:,:n_feat]) for ds in sms_osc]

hyper=Hyperalignment()
hypmaps=hyper([ds[:,:n_feat] for ds in zd_movie])

ha_ss_osc=[hypmaps[p].forward(ss_osc[p][:,:n_feat]) for p in np.arange(len(zd_movie))]
ha_sms_osc=[hypmaps[p].forward(sms_osc[p][:,:n_feat]) for p in np.arange(len(zd_movie))]

cv=CrossValidation(clf, NFoldPartitioner(attr='participants'), errorfx=mean_match_accuracy)

osc_bsc_ss=cv(vstack([ds[:,:n_feat] for ds in ss_osc]))
osc_bsc_sms=cv(vstack([ds[:,:n_feat] for ds in sms_osc]))


osc_bsha_ss=cv(vstack(ha_ss_osc))
osc_bsha_sms=cv(vstack(ha_sms_osc))

out_m=np.vstack([np.array([np.array(single_p).mean() for single_p in osc_wsc_ss]),np.array(osc_bsc_ss).flatten(1),np.array(osc_bsha_ss).flatten(1),np.array([np.array(single_p).mean() for single_p in osc_wsc_sms]),np.array(osc_bsc_sms).flatten(1), np.array(osc_bsha_sms).flatten(1)])
print time.time() - astart, out_m.mean(axis=0)
np.savetxt("./results/FL-MVPA/standard_space_results/"+mask_ID+".rstable.txt",out_m)
