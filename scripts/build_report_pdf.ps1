# Сборка PDF из report/report.md (Windows)
$root = Split-Path -Parent $PSScriptRoot
$md = Join-Path $root "report\report.md"
$pdf = Join-Path $root "report\report.pdf"

if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
    Write-Error "Установите Pandoc: https://pandoc.org/installing.html"
    exit 1
}

pandoc $md -o $pdf --from markdown --pdf-engine=xelatex -V geometry:margin=2.5cm
Write-Host "Saved: $pdf"
