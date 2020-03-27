
::python .\train_classfication.py -k Xception -e  500 -b 4 -s 256
::python .\train_classfication.py -k VGG -e  500 -b 4 -s 256
::python .\train_classfication.py -k ResNet34 -e  500 -b 4 -s 256
::python .\train_classfication.py -k Xception -e  500 -b 4 -s 256
python .\trainClass2input.py -k MyEncoder -e  500 -b 4 -s 256
::python .\trainClass2input.py -k DeeplabEncoder -e 500 -b 4 -s 256