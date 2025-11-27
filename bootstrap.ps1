Param(
  [string]$PythonBin = "python",
  [string]$VenvDir = ".venv"
)

Write-Host ">>> Using Python:" $PythonBin
& $PythonBin --version

# 创建虚拟环境
if (-not (Test-Path $VenvDir)) {
  Write-Host ">>> Creating venv at $VenvDir"
  & $PythonBin -m venv $VenvDir
}

# 激活虚拟环境
$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $activate

# 安装依赖
Write-Host ">>> Upgrading pip and installing dependencies"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

