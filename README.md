# xarm7_eye_control


```bash
conda env create --file environment.yml

conda activate xarm7_control
```


```bash
pip install -r requirements.txt

conda install pinocchio==3.2.0
```

### run

```bash
python main.py
```

### camera id

you can change the camera id in `main.py`:

```python
camera_id = 0  # change this to your camera id
```

or you can install pyrealsense2 to use realsense camera:

```bash
pip install pyrealsense2
```