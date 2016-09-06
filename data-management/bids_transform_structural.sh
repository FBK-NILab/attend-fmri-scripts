#!/bin/bash

if test -z $2
then
echo
echo "This script creates .nii image files from the storage"
echo "to your local directory and arranges the files in "
echo "a local ./data subdirectory in a BIDS-compliant way."
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
mkdir -p data

for s in $(seq 1 $key_file_length)
do
		
	sub_ID=$(cat $key_file | awk NR==$s | awk '{print $1}')
	sub_key=$(cat $key_file | awk NR==$s | awk '{print $2}')
	
		
	#echo $sub_key $sub_ID
	n=1
	for the_ses in $(ls -d $source_dir/*$sub_ID*)
	do

		echo $sub_key
		if [ $n == 1 ]
		then
			mkdir -p ./data/sub-$sub_key/ses-cogtasks/anat/
			mprage_dir=$(ls $the_ses/*/ | grep mprage)
			dcm2niix -b y -z y -o ./data/sub-$sub_key/ses-cogtasks/anat/ -f sub-$sub_key"_T1w" $the_ses/*/$mprage_dir
		else
			mkdir -p ./data/sub-$sub_key/ses-movie/anat/
			dcm2niix -b y -z y -o ./data/sub-$sub_key/ses-movie/anat/ -f sub-$sub_key"_T1w" -z y $the_ses/*/$mprage_dir

		fi
		n=$(echo $n+1 | bc)

	done
	#grep mprage | wc -w
	
	
done

fi

