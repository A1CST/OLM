@echo off
echo Setting up OLM CUDA Environment...
echo.

REM Create new conda environment
echo Creating conda environment 'olm_cuda'...
conda create -n olm_cuda python=3.11 -y

REM Activate the environment
echo Activating environment...
call conda activate olm_cuda

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

REM Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Environment setup complete!
echo To activate the environment, run: conda activate olm_cuda
echo To start the server, run: python web_server.py
echo.
pause 