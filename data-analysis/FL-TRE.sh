#!/bin/bash

# Extract Task - related Effects from Functional Localizer

curr_part=$1
curr_run=$2

func_loc_dir=./results/func-loc/$curr_part

input_data_dir=./results/preproc/$curr_part

curr_file_name=$curr_part"_task-osc_"$curr_run

input_data=$input_data_dir/ds.st.bet.moco.$curr_file_name"_bold.nii.gz"

mkdir -p $func_loc_dir

echo $curr_run

# Smooth the data
3dmerge -1blur_fwhm 6 -doall -prefix $func_loc_dir/$curr_file_name"_smooth".nii.gz $input_data
echo Smooth: DONE

# Scale the data in order to make them fluctuating around the mean
3dTstat -prefix $func_loc_dir/$curr_file_name"_mean".nii.gz $func_loc_dir/$curr_file_name"_smooth.nii.gz"
3dcalc -prefix $func_loc_dir/$curr_file_name"_scaled".nii.gz -a $func_loc_dir/$curr_file_name"_smooth.nii.gz" -b $func_loc_dir/$curr_file_name"_mean".nii.gz -c $input_data_dir/tmp.ds.st.bet.$curr_file_name"_bold_mask.nii.gz" -expr 'c*100*(a-b)/b'
echo Scaling: DONE


## Deconvolution

if [ $curr_run == "run-01" ]
        then
        3dDeconvolve -mask $input_data_dir/tmp.ds.st.bet.$curr_file_name"_bold_mask.nii.gz" -input $func_loc_dir/$curr_file_name"_scaled".nii.gz -nfirst 0 -polort 2 -GOFORIT 4 \
                -num_stimts 2 \
                -stim_times 1 '1D: 16 48 110 142 174 204 268 298' 'BLOCK(16,1)'  \
                -stim_label 1 intact \
                -stim_times 2 '1D: 32 64 94 126 188 220 252 284' 'BLOCK(16,1)' \
                -stim_label 2 scrambled \
                -num_glt 1 \
                -glt_label 1 IntvScr \
                -gltsym 'SYM: +intact -scrambled ' \
                -tout -x1D $func_loc_dir/$curr_file_name"_xmat.1D" -bucket $func_loc_dir/$curr_file_name"_stats"
elif [ $curr_run == "run-02" ]
        then
        3dDeconvolve -mask $input_data_dir/tmp.ds.st.bet.$curr_file_name"_bold_mask.nii.gz" -input $func_loc_dir/$curr_file_name"_scaled".nii.gz -nfirst 0 -polort 2 -GOFORIT 4 \
                -num_stimts 2 \
                -stim_times 1 '1D: 32 64 94 126 188 220 252 284' 'BLOCK(16,1)'  \
                -stim_label 1 intact \
                -stim_times 2 '1D: 16 48 110 142 174 204 268 298' 'BLOCK(16,1)' \
                -stim_label 2 scrambled \
                -num_glt 1 \
                -glt_label 1 IntvScr \
                -gltsym 'SYM: +intact -scrambled ' \
                -tout -x1D $func_loc_dir/$curr_file_name"_xmat.1D" -bucket $func_loc_dir/$curr_file_name"_stats"
fi

echo Deconvolution: DONE

## Use the structural image to align the results

### Extract coefficients from the contrasts

3dTcat -prefix $func_loc_dir/$curr_file_name"_intvscr.nii.gz" $func_loc_dir/$curr_file_name"_stats+orig[5]"
echo Extracting single coefficient: DONE


flirt -in $input_data_dir/$curr_part"_T1w_sess_ses-movie_bet.nii.gz" -ref $input_data -out $func_loc_dir/$curr_part"_T1w_ses-movie_in_"$curr_run

echo Transforming structural into functional: DONE

flirt -in $func_loc_dir/$curr_part"_T1w_ses-movie_in_"$curr_run -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz -omat $func_loc_dir/$curr_part"_struct_in_"$curr_run".transf.mat"

echo Transforming structural into functional into standard: DONE

flirt -in $func_loc_dir/$curr_file_name"_intvscr.nii.gz" -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz -applyxfm -init $func_loc_dir/$curr_part"_struct_in_"$curr_run".transf.mat" -out $func_loc_dir/$curr_file_name"_intvscr_inst"

echo Transforming structural into functional into standard: DONE
