# Neural_Simulation

export NSIMROOT=${PWD}



# Install Plasticinelab

Download mjpro150 and the activation key
https://roboti.us/download.html



```


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/issei/.mujoco/mjpro150/bin # maybe not needed

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

apt-get install -y mpich # need
sudo apt-get install patchelf

python -m pip install -e .

```