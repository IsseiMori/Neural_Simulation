## Install PyFLEX

### Note
Install PyFLEX in a directory close to the root as a path name that goes over 100 characters buffer will cause a buffer overflow error.


On local
```bash
conda install pybind11

git clone https://github.com/YunzhuLi/PyFleX.git

docker pull yunzhuli/pyflex_16_04_cuda_9_1

docker run \
  -v ${PWD}/PyFleX/:/workspace/PyFleX \
  -v /home/issei/anaconda3/envs/py37:/workspace/anaconda \
  -it yunzhuli/pyflex_16_04_cuda_9_1:latest

```

Inside Docker env
```
export PATH="/workspace/anaconda/bin:$PATH"
cd /workspace/PyFleX
export PYFLEXROOT=${PWD}
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
cd bindings; mkdir build; cd build; cmake ..; make -j
```

Back to local
```
cd PATH_TO_PyFleX
export PYFLEXROOT=${PWD}
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
cd ${PYFLEXROOT}/bindings/examples
python test_FluidFall.py
```

# Run FLEX data generate script
```
python datagen_flex.py --scene=RiceGrip --out="raw" --fixed_grip --video

```