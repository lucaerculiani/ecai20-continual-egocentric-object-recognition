Continual egocentric object recognition
==============

Code for the paper presented at [ECAI2020](https://arxiv.org/pdf/1912.05029.pdf)
The dataset pubished with the paper is available [here](https://ndownloader.figshare.com/files/17435471)

The code is provided togheter with a dockerfile, to ease the reproducibility of 
the experiments. The images were tesed under linux

How to run the code
------------

The steps:

 1. Install docker
 2. Build the image
 3. Run the container



After the installation of docker, clone the repository, open a terminal inside it and run

    $> docker build -t experiments_vm .
    
Then run the image (tested under linux on a machine with  64 GB of RAM)

    $> mkdir -p outputs ; docker run -v ${PWD}/inputs:/inputs -v ${PWD}/outputs:/outputs -v ${PWD}/info:/info  -v ${PWD}/results:/results -it  --shm-size=1G experiments_vm

The plots will appear in the folder `outputs/`

