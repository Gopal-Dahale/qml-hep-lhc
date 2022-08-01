# qml-hep-lhc

Edited this file on gcp

Format using `yapf -ir -vv .`

python training/train.py  -dc ElectronPhoton -mc ResnetV1 --epochs 2 -to-cat

python training/train.py  -dc ElectronPhoton -mc QCNNCong --epochs 3 -mm --pca 16 --batch-size 8 --use-quantum --feature-map AngleMap

python training/train.py -dc MNIST --binary-data "0 1" -cc 0.7 --resize "8 8" -mc QCNN --feature-map AmplitudeMap --ansatz Chen --cluster-state --n-layers 3 --drc -q -per-samp 0.03 --epochs 10 --batch-size 32
