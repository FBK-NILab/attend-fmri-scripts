#!/usr/bin/python

########################
#### Preamble
########################

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import time
astart=time.time()

from mvpa2.tutorial_suite import *

maskname=sys.argv[1]
maskfile=sys.argv[2]
mask_extent=int(sys.argv[3])
feature_list=[int(np.round(mask_extent*the_frac)) for the_frac in [0.25,0.50,0.75,0.85,0.90,0.95,1]]
print maskfile
print maskname
print feature_list

part_list=np.loadtxt("./pipeline/movie-part.list",dtype="str")
########################
#### Basic I/O
########################

pi_dataset=list()
vs_dataset=list()
mo_dataset=list()

n=0
curr_part=1
for p in part_list:


        # Perception - Imagery

        pi_data_list=np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-cogtasks_in_task-percim_run-*_tost.nii.gz"))
        pi_target_list=np.sort(glob.glob("./results/bids_beh/PI/"+p+".PI*.txt"))

        pi_sessions=pi_data_list.shape[0]
        pi_targets=pi_target_list.shape[0]


        pi_dataset.append(vstack([fmri_dataset(pi_data_list[i],mask=maskfile,targets=np.loadtxt(pi_target_list[i],dtype="str")[:,1]) for i in np.arange(pi_sessions)]))

        pi_dataset[n].sa['category']=np.hstack([np.loadtxt(pi_target_list[i],dtype="str")[:,1] for i in np.arange(pi_sessions)])
        pi_dataset[n].sa['task']=np.hstack([np.loadtxt(pi_target_list[i],dtype="str")[:,0] for i in np.arange(pi_sessions)])
        pi_dataset[n].sa['chunks']=np.hstack([np.repeat(i,170) for i in np.arange(1,pi_sessions+1)])
        pi_dataset[n].sa['participant']=list(np.repeat(curr_part,pi_dataset[n].shape[0]))

        ## Visual Search

        vs_data_list=np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-cogtasks_in_task-vs_run-*_tost.nii.gz"))
        vs_label_list=np.sort(glob.glob("./results/bids_beh/VS/labels/"+p+".VS*.txt"))
        vs_target_list=np.sort(glob.glob("./results/bids_beh/VS/delays/"+p+".VS*.txt"))

        vs_sessions=vs_data_list.shape[0]
        vs_targets=vs_target_list.shape[0]

        vs_dataset.append(vstack([fmri_dataset(vs_data_list[i],targets=np.loadtxt(vs_label_list[i],dtype="str")[5:],mask=maskfile) for i in np.arange(vs_sessions)]))
#       vs_dataset[n].targets=curr_vs_targets

        vs_dataset[n].sa['category']=np.hstack([np.loadtxt(vs_target_list[i],dtype="str")[5:,1] for i in np.arange(vs_sessions)])
        vs_dataset[n].sa['delay']=np.hstack([np.loadtxt(vs_target_list[i],dtype="str")[5:,2] for i in np.arange(vs_sessions)])
        vs_dataset[n].sa['chunks']=np.hstack([np.repeat(i,188) for i in np.arange(1,vs_sessions+1)])
        vs_dataset[n].sa['participant']=list(np.repeat(curr_part,vs_dataset[n].shape[0]))

        vs_dataset[n].sa['the_peaks']=np.hstack([np.loadtxt(vs_target_list[i],dtype="str")[5:,6] for i in np.arange(vs_sessions)]).astype("float").astype("int")
        vs_dataset[n].sa['single_peaks']=np.hstack([np.loadtxt(vs_target_list[i],dtype="str")[5:,5] for i in np.arange(vs_sessions)]).astype("float").astype("int")


        ## Movie (for Hyperalignment)

        movie_data_list=np.sort(glob.glob("./results/preproc/"+p+"/"+p+"_T1w_ses-movie_in_movie_run-*_tost.nii.gz"))
        movie_sessions=movie_data_list.shape[0]
        mo_dataset.append(vstack([fmri_dataset(movie_data_list[i],mask=maskfile) for i in np.arange(movie_sessions)]))
        mo_dataset[n].sa['chunks']=np.hstack([np.repeat(i,302) for i in np.arange(1,movie_sessions+1)])

#       print p,n,pi_sessions,pi_targets,vs_sessions,vs_targets

        print n
        n=n+1
        curr_part=curr_part+1

#       dataset[n].sa['chunks']=[np.repeat(i,170) for i in np.arange(1,nsessions)]


astop=time.time()
print astop-astart

########################
#### Sample Attributes
########################


for p in np.arange(len(pi_dataset)):

        # Perception / Imagery

        curr_p=np.argwhere(np.diff(1*(pi_dataset[p].sa.task=='P'))==1).flatten(1)+1
        curr_i=np.argwhere(np.diff(1*(pi_dataset[p].sa.task=='I'))==1).flatten(1)+1

        the_peaks=np.repeat('N',pi_dataset[p].shape[0])
        the_examples=np.zeros(pi_dataset[p].shape[0])
        single_peaks=np.zeros(pi_dataset[p].shape[0])

        n=1
        for v in curr_p:
                curr_idxs=np.arange(v+2,v+5)
                the_peaks[curr_idxs]=list(np.repeat('Y',3))
                the_examples[curr_idxs]=list(np.repeat(n,3))
                single_peaks[v+2]=1
                print n,v,curr_idxs,np.unique(np.array(pi_dataset[p][curr_idxs].sa['category']))[0]
                n=n+1

        n=1
        print "now c"
        for v in curr_i:
                curr_idxs=np.arange(v+2,v+5)
                print v,curr_idxs,np.unique(np.array(pi_dataset[p][curr_idxs].sa['category']))[0]
                the_peaks[curr_idxs]=list(np.repeat('Y',3))
                the_examples[curr_idxs]=list(np.repeat(n,3))
                single_peaks[v+2]=1
                n=n+1

        pi_dataset[p].sa['the_peaks']=the_peaks
        pi_dataset[p].sa['the_examples']=the_examples
        pi_dataset[p].sa['single_peaks']=single_peaks




##########################
###### Preprocessing
##########################

d_pi_dataset=list()
zd_pi_dataset=list()

d_vs_dataset=list()
zd_vs_dataset=list()

d_mo_dataset=list()
zd_mo_dataset=list()

for p in np.arange(len(pi_dataset)):
        detrender=PolyDetrendMapper(polyord=2,chunks_attr='chunks')

        detrender.train(pi_dataset[p])
        d_pi_dataset.append(pi_dataset[p].get_mapped(detrender))

        detrender.train(vs_dataset[p])
        d_vs_dataset.append(vs_dataset[p].get_mapped(detrender))

        detrender.train(mo_dataset[p])
        d_mo_dataset.append(mo_dataset[p].get_mapped(detrender))

        zscorer=ZScoreMapper(chunks_attr='chunks', param_est=('targets',['R']))

        zscorer.train(d_pi_dataset[p])
        zd_pi_dataset.append(d_pi_dataset[p].get_mapped(zscorer))

        zscorer.train(d_vs_dataset[p])
        zd_vs_dataset.append(d_vs_dataset[p].get_mapped(zscorer))

        mo_zscorer=ZScoreMapper(chunks_attr='chunks')
        mo_zscorer.train(d_mo_dataset[p])
        zd_mo_dataset.append(d_mo_dataset[p].get_mapped(mo_zscorer))




##########################
###### Sample Selection
##########################


#### Perception / Imagery

perc=[ds[ds.sa.task=='P'] for ds in zd_pi_dataset]
imag=[ds[ds.sa.task=='I'] for ds in zd_pi_dataset]

ss_perc=list()
ss_imag=list()

for p in np.arange(len(pi_dataset)):
        print p
        curr_pavg=np.zeros((np.max(perc[p].sa.the_examples),perc[p].shape[1]))
        for ex in np.arange(1,np.max(perc[p].sa.the_examples)+1):
                print ex
                curr_pavg[int(ex)-1,:]=perc[p][np.argwhere(perc[p].sa.the_examples==ex).flatten(1),:].samples.mean(axis=0)
        ss_perc.append(perc[p][np.array(perc[p].sa['single_peaks'])==1])
        ss_perc[p].samples=curr_pavg

        curr_iavg=np.zeros((np.max(imag[p].sa.the_examples),imag[p].shape[1]))
        for ex in np.arange(1,np.max(imag[p].sa.the_examples)+1):
                print ex
                curr_iavg[int(ex)-1,:]=imag[p][np.argwhere(imag[p].sa.the_examples==ex).flatten(1),:].samples.mean(axis=0)
        ss_imag.append(imag[p][np.array(imag[p].sa['single_peaks'])==1])
        ss_imag[p].samples=curr_iavg



ss_vs=list()


for p in np.arange(len(vs_dataset)):
#for p in [2,3,17]:

        example_list=np.unique(np.array(zd_vs_dataset[p].sa.single_peaks))[1:]
        curr_max_ex=np.unique(np.array(zd_vs_dataset[p].sa.single_peaks))[1:].shape[0]
        #np.max(np.array(zd_vs_dataset[p].sa.single_peaks))
        curr_vsavg=np.zeros((curr_max_ex,zd_vs_dataset[p].shape[1]))
#       print "I am here"
        n=0
        for ex in example_list:
#               print ex,n
#               print "I am inside the loop"
                curr_idxs=np.argwhere(zd_vs_dataset[p].sa.the_peaks==ex).flatten(1)
                curr_vsavg[n,:]=zd_vs_dataset[p][curr_idxs,:].samples.mean(axis=0)
                n=n+1
        ss_vs.append(zd_vs_dataset[p][np.array(zd_vs_dataset[p].sa['single_peaks'])>0])
        ss_vs[p].samples=curr_vsavg

# Feature Selection (based on correlation scores)

partlist=np.arange(len(mo_dataset))
totidxs=np.arange(len(partlist))

for p in np.arange(len(mo_dataset)):
        curr_part_list=totidxs[totidxs!=p]
        curr_corr_scores=np.zeros((zd_mo_dataset[p].shape[1],curr_part_list.shape[0]))
        for v in np.arange(curr_corr_scores.shape[0]):
                for k in np.arange(curr_part_list.shape[0]):
                        curr_corr_scores[v,k]=np.max([np.corrcoef(zd_mo_dataset[p].samples[:,v],zd_mo_dataset[curr_part_list[k]].samples[:,i])[0,1] for i in np.arange(zd_mo_dataset[curr_part_list[k]].shape[1])])
        print p
        curr_idxs=np.argsort(np.sum(curr_corr_scores,axis=1))
        zd_mo_dataset[p].fa['corr_scores']=np.sum(curr_corr_scores,axis=1)
        zd_mo_dataset[p].fa['corr_ranks']=np.argsort(np.sum(curr_corr_scores,axis=1))



#########################
##### Classification
##### C1 /C2 - Perception and Imagery
##### C3 - Perception / Preparatory Periods
##### C4 - Imagery / Preparatory Periods
##### C5 - Preparatory periods / Preparatory Periods
#########################




for nf in feature_list:
        fs_mo_dataset=[zd_mo_dataset[p][:,zd_mo_dataset[p].fa.corr_ranks[-nf:]] for p in np.arange(len(zd_mo_dataset))]
        fs_ss_perc=[ss_perc[p][:,zd_mo_dataset[p].fa.corr_ranks[-nf:]] for p in np.arange(len(zd_mo_dataset))]
        fs_ss_imag=[ss_imag[p][:,zd_mo_dataset[p].fa.corr_ranks[-nf:]] for p in np.arange(len(zd_mo_dataset))]
        fs_ss_vs=[ss_vs[p][:,zd_mo_dataset[p].fa.corr_ranks[-nf:]] for p in np.arange(len(zd_mo_dataset))]


        ### C1 / C2 - Perception and Imagery

        clf=LinearCSVMC()
        cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=mean_match_accuracy)

        ##### C12WSC

        p_wsc=[cv(ds) for ds in fs_ss_perc]
        i_wsc=[cv(ds) for ds in fs_ss_imag]


        ##### C12BSC

        cv=CrossValidation(clf,NFoldPartitioner(attr='participant'),errorfx=mean_match_accuracy)
        p_bsc=[cv(vstack(fs_ss_perc))]
        i_bsc=[cv(vstack(fs_ss_imag))]

        ##### C12BSHA

        hyper=Hyperalignment()
        hypmaps=hyper(fs_mo_dataset)

        ha_fs_ss_perc=[hypmaps[p].forward(fs_ss_perc[p]) for p in np.arange(len(fs_ss_perc))]
        ha_fs_ss_imag=[hypmaps[p].forward(fs_ss_imag[p]) for p in np.arange(len(fs_ss_imag))]

        p_bsha=[cv(vstack(ha_fs_ss_perc))]
        i_bsha=[cv(vstack(ha_fs_ss_imag))]


        ## Visual Search

        clf=LinearCSVMC()
        ptovs=np.zeros(len(vs_dataset))
        itovs=np.zeros(len(vs_dataset))
        ptovsha=np.zeros(len(vs_dataset))
        itovsha=np.zeros(len(vs_dataset))


        ha_fs_ss_vs=[hypmaps[p].forward(fs_ss_vs[p]) for p in np.arange(len(fs_ss_vs))]

        for p in np.arange(len(vs_dataset)):

                clf.train(fs_ss_perc[p])
                curr_ppred=clf.predict(fs_ss_vs[p])
                curr_ppredha=clf.predict(ha_fs_ss_vs[p])

                ptovs[p]=(np.sum(curr_ppred==fs_ss_vs[p].targets)/np.float(fs_ss_vs[p].targets.shape[0]))
                ptovsha[p]=(np.sum(curr_ppred==ha_fs_ss_vs[p].targets)/np.float(ha_fs_ss_vs[p].targets.shape[0]))




                clf.train(fs_ss_imag[p])
                curr_ipred=clf.predict(fs_ss_vs[p])
                curr_ipredha=clf.predict(ha_fs_ss_vs[p])
                itovs[p]=(np.sum(curr_ppred==fs_ss_vs[p].targets)/np.float(fs_ss_vs[p].targets.shape[0]))
                itovsha[p]=(np.sum(curr_ipredha==ha_fs_ss_vs[p].targets)/np.float(ha_fs_ss_vs[p].targets.shape[0]))

        out_m=np.vstack([np.vstack([[np.array(single_p).mean() for single_p in p_wsc],[np.array(single_i).mean() for single_i in i_wsc]]),p_bsc[0].samples.T,i_bsc[0].samples.T,p_bsha[0].samples.T,i_bsha[0].samples.T,ptovs,itovs,ptovsha,itovsha]).T
        np.savetxt("./"+maskname+".wHA."+str(nf)+".1D",out_m)


        print str(nf) + "DONE"
