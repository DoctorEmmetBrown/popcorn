import os
import glob

baseFolder = '/data/visitor/example/acquisition/folder/path/'
samples_list = ["*sample1*regular*expression*", "*sample2*regular*expression*", "*sample3*regular*expression*"]

for sample in samples_list:
    folders = glob.glob(baseFolder + '/*' + sample + '*')
    print(baseFolder + '/*' + sample + '*Propag')
    for folder in folders:
        if os.path.isdir(folder):
            print(folder)
            basename = os.path.basename(folder)
            par_file = folder + '/' + basename + 'pag0001.par'
            folderToCreate = baseFolder + '/volfloat/' + basename + 'pag'
            print(par_file)
            print(folderToCreate)
            if not (os.path.exists(folderToCreate)):
                os.mkdir(folderToCreate)
            cmd = 'PyHST2_2018a ' + par_file
            os.system(cmd)
