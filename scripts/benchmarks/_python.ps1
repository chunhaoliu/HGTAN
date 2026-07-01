function Resolve-BenchmarkPython {
    if ($env:ATUAV_PYTHON_EXE -and (Test-Path -LiteralPath $env:ATUAV_PYTHON_EXE)) {
        return $env:ATUAV_PYTHON_EXE
    }

    $pythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $pythonRoot) {
        $candidate = Get-ChildItem -LiteralPath $pythonRoot -Directory |
            Sort-Object Name -Descending |
            ForEach-Object {
                $exe = Join-Path $_.FullName "python.exe"
                if (Test-Path -LiteralPath $exe) { $exe }
            } |
            Select-Object -First 1
        if ($candidate) {
            return $candidate
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -and ($pythonCmd.Source -notmatch "WindowsApps")) {
        return $pythonCmd.Source
    }

    throw "Could not resolve a usable Python executable. Set ATUAV_PYTHON_EXE to python.exe."
}
