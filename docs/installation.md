To install mujoco follow: https://github.com/openai/mujoco-py.  

To get your computer id for the form on the mujoco website run the file `other/getid_linux`  
Probably also need to run these commands:  
- sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
- or: sudo apt-get install libglfw3 libglew2.0 (https://github.com/openai/mujoco-py/issues/549)
  
When mujoco is installed at folder `$HOME/.mujoco/mujoco200_linux` run this commands which adds variables to your ~/.bashrc script.  
```
echo -e "
export MUJOCO_GL=\"glfw\" \n\
export MJLIB_PATH=\$HOME/.mujoco/mujoco200_linux/bin/libmujoco200.so \n\
export MJKEY_PATH=\$HOME/.mujoco/mujoco200_linux/mjkey.txt \n\
export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco200_linux/bin:\$LD_LIBRARY_PATH\n\
export MUJOCO_PY_MJPRO_PATH=\$HOME/.mujoco/mujoco200_linux/ \n\
export MUJOCO_PY_MJKEY_PATH=\$HOME/.mujoco/mujoco200_linux/mjkey.txt \n\
" \
>>~/.bashrc

```