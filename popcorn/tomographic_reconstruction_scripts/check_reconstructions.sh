currentfolder=$(pwd)
for folders in */ ; do
    numfiles=$(ls $folders | wc -l)
    firstimage=$((numfiles / 3))
    secondimage=$((2 * numfiles / 3))
    echo "$secondimage"
    #for files in $folders*.edf ; do
    cd $folders
    firstfilename="$(find *$firstimage.edf)"
    secondfilename="$(find *$secondimage.edf)"
    cd ..
    cp "$currentfolder/$folders$firstfilename" "$currentfolder/reconstruction_check_folder/$firstfilename"
    cp "$currentfolder/$folders$secondfilename" "$currentfolder/reconstruction_check_folder/$secondfilename"
    echo "$firstfilename"
    echo "$secondfilename"
    #done
done
