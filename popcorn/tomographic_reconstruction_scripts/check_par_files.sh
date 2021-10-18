for folder in */
do
    numfiles=$(ls "$folder" | grep -c pag0001.par)

    if [[ $numfiles -gt 0 ]]
    then
        echo "$folder : $numfiles"
        #echo "OUI"
    else
        echo "$folder : No par file found."
        #echo "NON"
    fi
done