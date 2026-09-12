param(
    [string]$DatabaseUrl = $env:DATABASE_URL,
    [string]$OutputPath = (Join-Path $PSScriptRoot "ml-coach.backup")
)

if (-not $DatabaseUrl) {
    throw "请通过 DATABASE_URL 环境变量或 -DatabaseUrl 提供 PostgreSQL 连接串。"
}

if (-not (Get-Command pg_dump -ErrorAction SilentlyContinue)) {
    throw "未找到 pg_dump。请安装 PostgreSQL 客户端工具并将其加入 PATH。"
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

& pg_dump --format=custom --no-owner --no-privileges --file $OutputPath $DatabaseUrl
if ($LASTEXITCODE -ne 0) {
    throw "数据库备份失败，pg_dump 退出码：$LASTEXITCODE"
}

Write-Output "数据库备份完成：$OutputPath"