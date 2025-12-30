# GPU-Accelerated PDF RAG 启动脚本

Write-Host "🚀 启动 GPU 加速的 PDF RAG 应用..." -ForegroundColor Green

# 添加 Ollama 到 PATH
$env:Path += ";C:\Users\cools\AppData\Local\Programs\Ollama"

# 激活 conda 环境
Write-Host "`n📦 激活 rag_env 环境..." -ForegroundColor Yellow
conda activate rag_env

# 检查 GPU
Write-Host "`n🔍 检查 GPU 状态..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"未检测到GPU\"}')"

# 检查 Ollama 模型
Write-Host "`n🔍 检查 Ollama 模型..." -ForegroundColor Yellow
ollama list

Write-Host "`n🌐 启动 Streamlit 应用..." -ForegroundColor Green
Write-Host "访问: http://localhost:8501" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止服务`n" -ForegroundColor Yellow

# 启动应用
streamlit run pdf_rag_ui_ollama.py
