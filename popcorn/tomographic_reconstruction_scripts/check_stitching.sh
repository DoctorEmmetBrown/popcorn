currentfolder=$(pwd)
for folders in */ ; do
    numfiles=$(ls $folders | wc -l)
    if [[ "$folders" == *"00"* ]]; then
        #echo "not a stitching folder"
        num=2
    else
        if [[ "$folders" == *"decomposition"* ]]; then
            num=3
        else
            echo "$folders, $numfiles files"
        fi
    fi
    #done
done
