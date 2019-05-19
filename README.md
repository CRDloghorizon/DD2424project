# DD2424project
DD2424 project on ShuffleNet using PyTorch

## Prepare Dataset
1. Download the images from http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
  
## Train Example
  ```bash
  python trainv2.py -a 'shufflenetv2' -b batch_size --epochs n_epochs --wd 4e-5 --lr 0.1 --ratio network_ratio /folder/to/imagenet/
  ```
  
## Evaluate Example
  ```bash
  python trainv2.py -a 'shufflenetv2' -b 8 --ratio network_ratio --evaluate 'path to model' /folder/to/imagenet/
  ```
