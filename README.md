# qml-hep-lhc

Format using `yapf -ir -vv .`

python training/train.py  -dc ElectronPhoton -mc ResnetV1 --epochs 2 -to-cat

python training/train.py  -dc ElectronPhoton -mc QCNNCong --epochs 3 -mm --pca 16 --batch-size 8 --use-quantum --feature-map AngleMap