## Deployment

1. Download deployment kit

    * Version 1.0.5
    
        * [For Ubuntu 18.04 with x84_64 Architecture](https://drive.google.com/drive/folders/1nCHiryGM2iKKRjd1dTPQvEnR0OPjvhRw?usp=sharing)
        * [For Ubuntu 18.04 with ppc64le Architecture](https://drive.google.com/drive/folders/1k-NdSmwJkCyyMBBvOICSWhD24C2ZvW_Q?usp=sharing)
        * [For CentOS 7 with x86_64 Architecture](https://drive.google.com/drive/folders/1tOte_s8FmdNu0FqpXgJoUY_fFo7tcTNc?usp=sharing)

    * Version 1.0.4
    
        * [For Ubuntu 18.04 with x84_64 Architecture](https://drive.google.com/drive/folders/1SC9gvZFiQRcNNd5YydMqXtgmD7TVVvbp?usp=sharing)
        * [For Ubuntu 18.04 with ppc64le Architecture](https://drive.google.com/drive/folders/1lkkVvIYPnKG7wuzukk41SB6_lFL20Dp5?usp=sharing)
        * [For CentOS 7 with x86_64 Architecture](https://drive.google.com/drive/folders/1SC9gvZFiQRcNNd5YydMqXtgmD7TVVvbp?usp=sharing)

1. Extract deployment kit and then switch to extracted directory which should have following structure: 
    
    ```
    - mirle-vision-deployment-kit-<os>-<arch>
        cuda_10.2.xxx.run
        cudnn-10.2-linux-xxx-v7.6.xxx.tgz
        Miniconda3-latest-Linux-<arch>.sh 
        checkpoints.zip
        mirle-vision.pytorch.zip
        mvision.tar.gz
    ```

1. Install `CUDA` (including `NVIDIA Driver`) and `cuDNN`

    * For CentOS 7 with x86_64 Architecture
    
        * Install development tools
        
            ```
            $ yum groupinstall "Development Tools"
            $ yum install kernel-devel epel-release
            $ reboot
            ```
          
            > If the internet is unreachable, try to install from media (CD-ROM or USB) repository, note that this requires consistent kernel version between OS and media one.
            > ```
            > $ lsblk  # to check media block, e.g.: sdb1
            > $ mkdir -p /media/cdrom
            > $ mount /dev/sdb1 /media/cdrom
            > 
            > Then you are able to enable media repository by append `--enablerepo=c7-media` to `yum` command, e.g.:
            > $ yum --enablerepo=c7-media groupinstall "Development Tools"
            > $ yum --enablerepo=c7-media install kernel-devel epel-release
            > ```
        
        * Disable `Nouveau` driver
        
            ```
            $ lshw -numeric -C display | grep nouveau  # to check `Nouveau` driver is enabled (expected something is shown)
            $ vim /etc/default/grub
            Append `nouveau.modeset=0` to `GRUB_CMDLINE_LINUX`
          
            Apply to `GRUB` for BIOS or UEFI
            $ sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # for BIOS
            $ sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg  # for UEFI
            
            $ reboot
            $ lshw -numeric -C display | grep nouveau  # to check `Nouveau` driver is disabled (expected nothing is shown)
            ```
          
        * Disable `X Server`
        
            ```
            $ systemctl isolate multi-user.target
            ```
          
        * Install `NVIDIA Driver` and `CUDA 10.2`
        
            ```
            $ ./cuda_10.2.xxx.run
            No DKMS
            No OpenGL
            No Samples
            
            $ vim ~/.bashrc
            Append `export CUDA_HOME=/usr/local/cuda`
            Append `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
            Append `export PATH=$PATH:/usr/local/cuda/bin`
            $ source ~/.bashrc
            $ nvidia-smi  # to check if CUDA installed successfully
            ```
          
        * Install `cuDNN 7.6`
        
            ```
            $ tar -zxvf cudnn-10.2-linux-xxx-v7.6.xxx.tgz
            $ cp cuda/include/* /usr/local/cuda/include/
            $ cp cuda/lib64/* /usr/local/cuda/lib64/
            ```
          
        * Enable `X Server`
        
            ```
            $ systemctl start graphical.target
            ```

1. Extract `mirle-vision.pytorch`

    ```
    $ unzip mirle-vision.pytorch.zip -d ~/ 
    ```
   
1. Put pre-downloaded checkpoints

    ```
    $ mkdir -p ~/.cache/torch/hub
    $ unzip checkpoints.zip -d ~/.cache/torch/hub/
    ```

1. Install `Miniconda3`

    ```
    $ bash Miniconda3-latest-Linux-<arch>.sh
    by running conda init? => choose yes
    $ source ~/.bashrc
    ```
   
1. Setup virtual environment 

    ```
    $ tar -xvf mvision.tar.gz -C ~/miniconda3/envs/mvision
    $ source ~/miniconda3/envs/mvision/bin/activate
    ```
    
    > To verify PyTorch works well with CUDA and cuDNN
    > ```
    > (mvision) $ python
    >           > import torch
    >           > torch.tensor(1).cuda() 
    >           > torch.backends.cudnn.enabled
    > ```

1. Install `mirle-vision.pytorch`

    ```
    (mvision) $ cd ~/mirle-vision.pytorch
              $ pip install -e .
    ```
    
    > To verify library works well
    > ```
    > (mvision) $ python
    >           > import aibox_vision.api
    >           > import aibox_vision.lib
    > ```

1. (Optional) Compile all python scripts

    ```
    $ python -m compileall -b ./src
    $ find ./src -name '*.py' -exec rm {} \;
    ```

1. Launch web service

    ```
    $ mkdir ~/mirle-vision.pytorch/envs
    $ ln -s ~/miniconda3/envs/mvision ~/mirle-vision.pytorch/envs/mvision
    $ python bin/create_launch_prod.py
    $ bash bin/launch-prod.sh
    ```

    > To verify web service has launched successfully
    > ```
    > $ curl -X GET http://127.0.0.1:5000/api
    > ```

1. Make web service automatically launch on boot

    > Go on after web service has stopped

    ```
    $ python bin/create_service.py
    $ mv mirle-vision.service /etc/systemd/system/mirle-vision.service
    $ systemctl daemon-reload
    $ systemctl enable mirle-vision.service
    $ systemctl start mirle-vision.service
    $ systemctl status mirle-vision.service  # for check short logs
    $ journalctl -u mirle-vision.service # for check full logs
    ```
    
    > To verify automatically launch web service works well, reboot and check
    > ```
    > $ curl -X GET http://127.0.0.1:5000/api
    > ```
