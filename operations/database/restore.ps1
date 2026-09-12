param(
    [Parameter(Mandatory)]
    [string]$BackupPath,
    [string]$DatabaseUrl = $env:DATABASE_URL
)

if (-not $DatabaseUrl) {
    throw "请通过 DATABASE_URL 环境变量或 -DatabaseUrl 提供 PostgreSQL 连接串。"
}

if (-not (Test-Path -LiteralPath $BackupPath -PathType Leaf)) {
    throw "备份文件不存在：$BackupPath"
}

if (-not (Get-Command pg_restore -ErrorAction SilentlyContinue)) {
    throw "未找到 pg_restore。请安装 PostgreSQL 客户端工具并将其加入 PATH。"
}

& pg_restore --clean --if-exists --no-owner --no-privileges --dbname $DatabaseUrl $BackupPath
if ($LASTEXITCODE -ne 0) {
    throw "数据库恢复失败，pg_restore 退出码：$LASTEXITCODE"
}

Write-Output "数据库恢复完成。"