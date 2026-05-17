# 等待全量扫描进程结束后，按当日报告发邮件
param(
    [int]$WaitPid,
    [string]$DateTag = (Get-Date -Format "yyyyMMdd")
)
$ErrorActionPreference = "Stop"
Set-Location "d:\efi"
Write-Host "等待扫描进程 PID=$WaitPid 结束..."
Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host "开始发送邮件（报告日期 $DateTag）..."
py -3 -u volume\volume_ma_filter_daily_all.py --send-email-only --date $DateTag
