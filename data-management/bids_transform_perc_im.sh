#!/bin/bash

if test -z $2
then
echo
echo "This script creates .nii image files from the storage to your local directory and arranges the files a local ./data subdirectory in a BIDS-compliant way."
echo
echo "You should run it from your working directory in the following way:"
echo
echo "./<path-to-the-script>/transform_structural.sh <path-to-cimec-storage2-mounted-directory> <path-to-the-keys-file>"
echo
echo "where <cimec-storage2-mounted-directory> refers to the path where you mounted source files from cimec-storage2 to your local environment."
else

source_dir=$1
key_file=$2

key_file_length=$(cat $key_file | wc -l)
#echo $key_file_length


for s in $(seq 1 $key_file_length)
do
		
	sub_ID=$(cat $key_file | awk NR==$s | awk '{print $1}')
	sub_key=$(cat $key_file | awk NR==$s | awk '{print $2}')
#	echo $sub_ID $sub_key

	run_counter=1

	for the_run in $(ls $source_dir/*$sub_ID*/*/ |grep PI_DiCo)
	do

		

		n_files=$(ls $source_dir/*$sub_ID*/*/$the_run/*.DCM | wc -w)
#		echo $n_files
		if [ $n_files == 175 ]
		then
			mkdir -p data/sub-$sub_key/ses-cogtasks/func
#			echo $sub_ID $sub_key $the_run r0$run_counter
			
		   	dcm2niix -b y -o ./data/sub-$sub_key/ses-cogtasks/func -f sub-$sub_key"_task-percim_run-0"$run_counter"_bold" -z y $source_dir/*$sub_ID*/*/$the_run

			run_counter=$(echo $run_counter +1 | bc)
			

		fi
		


	done

done

fi
