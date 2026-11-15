1. Install python virtual environment
sudo apt update
sudo apt install python3.11-venv

2. Go to project's director
cd ~/Development/iris-recognition

3. Create the virtual environment
python3.11 -m venv .venv

4. Active the virtual environment
source .venv/bin/activate

5. Inside environment, install Open Iris via pip
python -m pip install --upgrade pip
pip install open-iris

6. Run scripts inside /src directory
-python -m src.create-templates
-python -m src.matching
-python -m src.identify-by-image dataset/110/L/S1110L01.jpg --eye left --top 5
-python src.identify-by-subject.py dataset/012/R/S1012R02.jpg --eye right --top 5

** problem with ONNX Runtime library
-cp .venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so .venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.backup.so
-patchelf --print-execstack .venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so
-patchelf --clear-execstack .venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so
