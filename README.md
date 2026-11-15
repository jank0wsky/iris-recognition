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

