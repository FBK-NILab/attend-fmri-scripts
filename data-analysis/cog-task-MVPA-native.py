#!/usr/bin/python

import numpy as np
import sys


import glob
import time
astart=time.time()

from mvpa2.tutorial_suite import *

mask_ID=sys.argv[1]



part_list=np.loadtxt("./results/func-loc/osc-part-list-mod.txt",dtype="str")


pidatalist=list()
vsdatalist=list()
modatalist=list()

n=0
for p in part_list:

#       curr_mask_name="./results/func-loc/"+p+"/clusters/"+mask_ID+".in."+p+".nii.gz"

        curr_mask_name="./results/classtrial/fl-native-space/native-space-clusters/"+p+"/"+mask_ID+".in."+p+".nii.gz"

        currmofiles=(np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-movie_in_movie_run-0*_tonativesp.nii.gz")))

        currpifiles=(np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-cogtasks_in_task-percim_run-0*_tonativesp.nii.gz")))

#       currvsfiles=(np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-cogtasks_in_task-vs_run-0*_tonativesp.nii.gz")))


#       print currmofiles,currpifiles,currvsfiles
        print "Start "+str(n)

        # Perception and Imagery

        pitargets=np.vstack([np.loadtxt('./results/bids_beh/PI/'+p+'.PI.r0'+str(i)+'.txt',dtype='str') for i in np.arange(1,currpifiles.shape[0]+1)])

        pidatalist.append(vstack([fmri_dataset(currpifiles[i-1], mask=curr_mask_name, targets=np.loadtxt('./results/bids_beh/PI/'+p+'.PI.r0'+str(i)+'.txt',dtype='str')[:,1]) for i in np.arange(1,currpifiles.shape[0]+1)]))
#pitargets[np.arange((i*170),((i*170)+170))]) for i in np.arange(currpifiles.shape[0])]))

        pidatalist[n].sa['category']=list(pitargets[:,1])
        pidatalist[n].sa['task']=list(pitargets[:,0])
        pidatalist[n].sa['chunks']=np.hstack([np.repeat(i,170) for i in np.arange(1,currpifiles.shape[0]+1)])
        pidatalist[n].sa['participant']=list(np.repeat(n+1,170*currpifiles.shape[0]))

        pidatalist[n].sa['singlesample']=list(np.loadtxt("./results/bids_beh/PI/the_peaks/"+p+".PI.peaks.txt")[:,1])
        pidatalist[n].sa['smoothsamples']=list(np.loadtxt("./results/bids_beh/PI/the_peaks/"+p+".PI.peaks.txt")[:,2])



        # Visual Search
#       vsdatalist.append(vstack([fmri_dataset(currvsfiles[i-1],mask=curr_mask_name,
#               targets=np.loadtxt('./results/bids_beh/VS/labels/'+p+'.VS.labels.r0'+str(i)+'.txt',dtype='str')[5:]) for i in np.arange(1,currvsfiles.shape[0]+1) ]))

#       vsdatalist[n].sa['category']=vsdatalist[n].targets
#       vsdatalist[n].sa['delay']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,1])
#       vsdatalist[n].sa['chunks']=np.hstack([np.repeat(i,188) for i in np.arange(1,currvsfiles.shape[0]+1)])
#       vsdatalist[n].sa['participant']=list(np.repeat(n+1,vsdatalist[n].shape[0]))

#       vsdatalist[n].sa['firstsample']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,2])
#
#       vsdatalist[n].sa['lastsample']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,3])

#       vsdatalist[n].sa['singlesample']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,4])
#
#       vsdatalist[n].sa['smoothsamples']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,5])

#       vsdatalist[n].sa['allsamples']=list(np.loadtxt("./results/bids_beh/VS/the_peaks/"+p+".VS.peaks.txt")[:,6])

#       print "VisS: END"

        # Movie
        modatalist.append(vstack([fmri_dataset(currmofiles[i],mask=curr_mask_name) for i in np.arange(6)]))
        modatalist[n].sa['chunks']=np.hstack([np.repeat(i,302) for i in np.arange(1,7)])
        modatalist[n].sa['participant']=list(np.repeat(n+1,modatalist[n].shape[0]))

        print n,"done"
        print time.time() - astart, p
        n=n+1

d_pi=list()
zd_pi=list()

#d_vs=list()
#zd_vs=list()

d_movie=list()
zd_movie=list()

### Preprocessing

for p in np.arange(len(modatalist)):

        detrender=PolyDetrendMapper(polyord=2,chunks_attr='chunks')

        detrender.train(pidatalist[p])
        d_pi.append(pidatalist[p].get_mapped(detrender))

#       detrender.train(vsdatalist[p])
#       d_vs.append(vsdatalist[p].get_mapped(detrender))

        detrender.train(modatalist[p])
        d_movie.append(modatalist[p].get_mapped(detrender))

        zscorer=ZScoreMapper(chunks_attr='chunks')

        zscorer.train(d_pi[p])
        zd_pi.append(d_pi[p].get_mapped(zscorer))

#       zscorer.train(d_vs[p])
#       zd_vs.append(d_vs[p].get_mapped(zscorer))

        zscorer.train(d_movie[p])
        zd_movie.append(d_movie[p].get_mapped(zscorer))




### Sample Selection

##### Single Samples

ss_perc=[ds[np.logical_and(ds.sa.task=="P",ds.sa.singlesample!=0)] for ds in zd_pi]
ss_imag=[ds[np.logical_and(ds.sa.task=="I",ds.sa.singlesample!=0)] for ds in zd_pi]
#ss_vs=[ds[ds.sa.firstsample!=0] for ds in zd_vs]
#fs_vs=[ds[ds.sa.lastsample!=0] for ds in zd_vs]
#ls_vs=[ds[ds.sa.singlesample!=0] for ds in zd_vs]

#### Smooth samples

#sms_perc=list()
#sms_imag=list()
#sms_vs=list()
#asm_vs=list()


## Perception and Imagery
##for p in np.arange(len(zd_movie)):

##      curr_avg=np.zeros((,zd_osc[p].shape[1]))
##      for t in np.unique(trial_ID)[1:]:
##              print t
##              curr_avg[t-1,:]=zd_osc[p].samples[np.argwhere(trial_ID==t).flatten(1),:].mean(axis=0)
##      ds=ss_osc[p].copy(deep=False,sa=['targets', 'chunks','participants'],fa=['voxel_indices'])
##      ds.samples=curr_avg
##      sms_osc.append(ds)

### Classification

min_feat=int(np.round(np.min([zd_movie[p].shape[1] for p in np.arange(len(zd_movie))])*.75))

# Rough feature selection

clf=LinearCSVMC()
#cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=mean_match_accuracy)

#perc_ss_wsc=[cv(ds[:,:min_feat]) for ds in ss_perc]
#imag_ss_wsc=[cv(ds[:,:min_feat]) for ds in ss_imag]

# Perc_Im classification
perc_to_im_mat=np.zeros((len(pidatalist),3))

for p in np.arange(len(pidatalist)):
        perc_to_im_mat[p,0]=int(part_list[p][4:6])
        clf.train(ss_perc[p])
        perc_to_im_mat[p,1]=np.mean(clf.predict(ss_imag[p])==ss_imag[p].targets)
        clf.train(ss_imag[p])
        perc_to_im_mat[p,2]=np.mean(clf.predict(ss_perc[p])==ss_perc[p].targets)






#vs_ss_wsc=[cv(ds[:,:min_feat]) for ds in ss_vs]
#vs_fs_wsc=[cv(ds[:,:min_feat]) for ds in fs_vs]
#vs_ls_wsc=[cv(ds[:,:min_feat]) for ds in ls_vs]


#hyper=Hyperalignment(alignment=ProcrusteanMapper(svd='dgesvd',space='commonspace'))
#hypmaps=hyper([ds[:,:min_feat] for ds in zd_movie])

#ha_ss_perc=[hypmaps[p].forward(ss_perc[p][:,:min_feat]) for p in np.arange(len(zd_movie))]
#ha_ss_imag=[hypmaps[p].forward(ss_imag[p][:,:min_feat]) for p in np.arange(len(zd_movie))]

#ha_ss_vs=[hypmaps[p].forward(ss_vs[p][:,:min_feat]) for p in np.arange(len(zd_movie))]
#ha_fs_vs=[hypmaps[p].forward(fs_vs[p][:,:min_feat]) for p in np.arange(len(zd_movie))]
#ha_ls_vs=[hypmaps[p].forward(ls_vs[p][:,:min_feat]) for p in np.arange(len(zd_movie))]


#cv=CrossValidation(clf, NFoldPartitioner(attr='participant'), errorfx=mean_match_accuracy)

#perc_to_ss_vs=list()
#perc_to_fs_vs=list()
#perc_to_ls_vs=list()

#imag_to_ss_vs=list()
#imag_to_fs_vs=list()
#imag_to_ls_vs=list()


#for p in np.arange(len(vsdatalist)):

#       clf.train(ss_perc[p])
#       perc_to_ss_vs.append(np.float(np.sum(clf.predict(ss_vs[p])==ss_vs[p].targets))/ss_vs[p].targets.shape[0])
#       perc_to_fs_vs.append(np.float(np.sum(clf.predict(fs_vs[p])==fs_vs[p].targets))/fs_vs[p].targets.shape[0])
#       perc_to_ls_vs.append(np.float(np.sum(clf.predict(ls_vs[p])==ls_vs[p].targets))/ls_vs[p].targets.shape[0])

#       clf.train(ss_imag[p])
#       imag_to_ss_vs.append(np.float(np.sum(clf.predict(ss_vs[p])==ss_vs[p].targets))/ss_vs[p].targets.shape[0])
#       imag_to_fs_vs.append(np.float(np.sum(clf.predict(fs_vs[p])==fs_vs[p].targets))/fs_vs[p].targets.shape[0])
#       imag_to_ls_vs.append(np.float(np.sum(clf.predict(ls_vs[p])==ls_vs[p].targets))/ls_vs[p].targets.shape[0])

#perc_ss_bsha=cv(vstack(ha_ss_perc))
#imag_ss_bsha=cv(vstack(ha_ss_imag))

#vs_ss_bsha=cv(vstack(ha_ss_vs))
#vs_fs_bsha=cv(vstack(ha_fs_vs))
#vs_ls_bsha=cv(vstack(ha_ls_vs))


#vs_out_m=np.vstack([[np.array(single_p).mean() for single_p in vs_ss_wsc],np.array(vs_ss_bsha).flatten(1),[np.array(single_p).mean() for single_p in vs_fs_wsc],np.array(vs_fs_bsha).flatten(1),[np.array(single_p).mean() for single_p in vs_ls_wsc],np.array(vs_ls_bsha).flatten(1)])
#perc_imag_out_m=np.vstack([[np.array(single_p).mean() for single_p in perc_ss_wsc],np.array(perc_ss_bsha).flatten(1),[np.array(single_p).mean() for single_p in imag_ss_wsc],np.array(imag_ss_bsha).flatten(1)])

#cross_modality=np.vstack([perc_to_fs_vs,perc_to_ss_vs,perc_to_ls_vs,imag_to_fs_vs,imag_to_ss_vs,imag_to_ls_vs])

np.savetxt("./results/classtrial/fl-native-space/percim-class-results/"+mask_ID+".percimtable.txt",perc_to_im_mat)

print time.time() - astart
#np.savetxt("./results/classtrial/fl-native-space/fl-class-results/"+mask_ID+".pitable.txt",perc_imag_out_m)
#np.savetxt("./results/classtrial/fl-native-space/fl-class-results/"+mask_ID+".vstable.txt",vs_out_m)
#np.savetxt("./results/classtrial/fl-native-space/fl-class-results/"+mask_ID+".cstable.txt",cross_modality)
