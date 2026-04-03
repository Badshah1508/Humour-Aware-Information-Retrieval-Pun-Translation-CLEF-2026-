param(
    [ValidateSet("laptop", "fast", "full")]
    [string]$Profile = "fast",

    [ValidateSet("train", "dense", "rerank", "all")]
    [string]$Step = "all",

    [switch]$Benchmark,

    [switch]$WriteBenchmarkCsv,

    [switch]$CompareBenchmark,

    [ValidateRange(1, 20)]
    [int]$TopN = 3
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Profile: $Profile | Step: $Step"

$timings = [ordered]@{}
$startedAt = Get-Date
$pythonCmd = "python"

function Resolve-PythonCommand {
    $venvPython = Join-Path $repoRoot "venv/Scripts/python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return "python"
}

function Show-BenchmarkComparison {
    $csvPath = Join-Path $repoRoot "logs/runtime_benchmark.csv"
    if (-not (Test-Path $csvPath)) {
        Write-Host "[compare] no benchmark CSV found at: $csvPath"
        Write-Host "[compare] first run with: -Benchmark -WriteBenchmarkCsv"
        return
    }

    $rows = @(Import-Csv -Path $csvPath)
    if (-not $rows -or $rows.Count -eq 0) {
        Write-Host "[compare] benchmark CSV exists but has no rows."
        return
    }

    Write-Host ""
    Write-Host "===== Benchmark Comparison ====="
    Write-Host "source: $csvPath"
    Write-Host ("entries: {0}" -f $rows.Count)
    Write-Host ("topN: {0}" -f $TopN)

    $latest = $rows | Sort-Object timestamp -Descending | Select-Object -First 1
    Write-Host ("latest: {0} | profile={1} | step={2} | total={3} sec" -f $latest.timestamp, $latest.profile, $latest.step, $latest.total_sec)
    Write-Host ""

    $stages = @("dense_sec", "train_sec", "rerank_sec", "total_sec")
    foreach ($stage in $stages) {
        $valid = $rows | Where-Object { $_.$stage -and [double]::TryParse($_.$stage, [ref]([double]0)) }
        if (-not $valid -or $valid.Count -eq 0) {
            continue
        }

        $best = $valid | Sort-Object { [double]$_.$stage } | Select-Object -First 1
        $value = [math]::Round([double]$best.$stage, 2)
        Write-Host ("best {0}: {1} sec | profile={2} | step={3} | machine={4} | at={5}" -f $stage, $value, $best.profile, $best.step, $best.machine, $best.timestamp)

        $topRows = @($valid | Sort-Object { [double]$_.$stage } | Select-Object -First $TopN)
        if ($topRows.Count -gt 0) {
            Write-Host ("top {0} for {1}:" -f $TopN, $stage)
            $rank = 1
            foreach ($row in $topRows) {
                $stageValue = [math]::Round([double]$row.$stage, 2)
                Write-Host ("  {0}. {1} sec | profile={2} | step={3} | machine={4} | at={5}" -f $rank, $stageValue, $row.profile, $row.step, $row.machine, $row.timestamp)
                $rank += 1
            }
        }
    }

    Write-Host "==============================="
}

function Set-CommonEnv {
    $env:PYTHONUNBUFFERED = "1"
    $env:TOKENIZERS_PARALLELISM = "false"
    # Prevent external Python distributions (e.g., Anaconda) from contaminating venv imports
    $env:PYTHONHOME = ""
    $env:PYTHONPATH = ""
}

function Set-LaptopProfile {
    # Conservative profile for low-memory CPU laptops (e.g., 2C/4T, iGPU)
    $env:CE_EPOCHS = "1"
    $env:CE_BATCH_SIZE = "4"
    $env:CE_EVAL_STEPS = "2000"
    $env:CE_MAX_HARD_NEGS = "5"
    $env:CE_MAX_RANDOM_NEGS = "2"
    $env:CE_NUM_WORKERS = "0"
    $env:CE_PIN_MEMORY = "false"

    $env:DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    $env:DENSE_BATCH_SIZE = "16"
    $env:DENSE_TOP_K = "30"
    $env:DENSE_EXPERIMENT_TOP_K = "30"

    $env:CROSS_ENCODER_CANDIDATES = "10"
}

function Set-FastProfile {
    # Fast profile for typical CPU or entry-level GPU machines
    $env:CE_EPOCHS = "2"
    $env:CE_BATCH_SIZE = "6"
    $env:CE_EVAL_STEPS = "1000"
    $env:CE_MAX_HARD_NEGS = "8"
    $env:CE_MAX_RANDOM_NEGS = "3"
    $env:CE_NUM_WORKERS = "0"
    $env:CE_PIN_MEMORY = "false"

    $env:DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    $env:DENSE_BATCH_SIZE = "32"
    $env:DENSE_TOP_K = "40"
    $env:DENSE_EXPERIMENT_TOP_K = "40"

    $env:CROSS_ENCODER_CANDIDATES = "15"
}

function Set-FullProfile {
    $env:CE_EPOCHS = "8"
    $env:CE_BATCH_SIZE = "16"
    $env:CE_EVAL_STEPS = "200"
    $env:CE_MAX_HARD_NEGS = "40"
    $env:CE_MAX_RANDOM_NEGS = "10"
    $env:CE_NUM_WORKERS = "2"
    $env:CE_PIN_MEMORY = "auto"

    $env:DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    $env:DENSE_BATCH_SIZE = "128"
    $env:DENSE_TOP_K = "100"
    $env:DENSE_EXPERIMENT_TOP_K = "100"

    $env:CROSS_ENCODER_CANDIDATES = "100"
}

function Run-Train {
    Write-Host "Running cross-encoder training..."
    & $pythonCmd -m src.retrieval.train_cross_encoder
}

function Run-Dense {
    Write-Host "Running dense retrieval..."
    & $pythonCmd -m src.retrieval.run_dense
}

function Run-Rerank {
    Write-Host "Running cross-encoder reranking..."
    & $pythonCmd -m src.retrieval.run_cross_encoder
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $stepStart = Get-Date
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }

    $seconds = [math]::Round(((Get-Date) - $stepStart).TotalSeconds, 2)
    $timings[$Name] = $seconds

    if ($Benchmark) {
        Write-Host "[benchmark] ${Name}: $seconds sec"
    }
}

function Write-BenchmarkSummary {
    $totalSeconds = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 2)
    $totalMinutes = [math]::Round($totalSeconds / 60, 2)

    Write-Host ""
    Write-Host "===== Runtime Summary ====="
    Write-Host "Profile: $Profile"
    Write-Host "Step: $Step"

    foreach ($k in $timings.Keys) {
        Write-Host ("- {0}: {1} sec" -f $k, $timings[$k])
    }

    Write-Host ("- total: {0} sec ({1} min)" -f $totalSeconds, $totalMinutes)
    Write-Host "==========================="

    if ($WriteBenchmarkCsv) {
        $logsDir = Join-Path $repoRoot "logs"
        if (-not (Test-Path $logsDir)) {
            New-Item -ItemType Directory -Path $logsDir | Out-Null
        }

        $csvPath = Join-Path $logsDir "runtime_benchmark.csv"

        $dense = if ($timings.Contains("dense")) { $timings["dense"] } else { "" }
        $train = if ($timings.Contains("train")) { $timings["train"] } else { "" }
        $rerank = if ($timings.Contains("rerank")) { $timings["rerank"] } else { "" }

        $record = [PSCustomObject]@{
            timestamp   = (Get-Date).ToString("s")
            profile     = $Profile
            step        = $Step
            dense_sec   = $dense
            train_sec   = $train
            rerank_sec  = $rerank
            total_sec   = $totalSeconds
            total_min   = $totalMinutes
            machine     = $env:COMPUTERNAME
        }

        if (Test-Path $csvPath) {
            $record | Export-Csv -Path $csvPath -NoTypeInformation -Append
        } else {
            $record | Export-Csv -Path $csvPath -NoTypeInformation
        }

        Write-Host "[benchmark] saved: $csvPath"
    }
}

if ($CompareBenchmark) {
    Show-BenchmarkComparison
} else {
    $pythonCmd = Resolve-PythonCommand
    Write-Host "Python: $pythonCmd"

    Set-CommonEnv
    if ($Profile -eq "laptop") {
        Set-LaptopProfile
    } elseif ($Profile -eq "fast") {
        Set-FastProfile
    } else {
        Set-FullProfile
    }

    switch ($Step) {
        "train" { Invoke-Step -Name "train" -Action ${function:Run-Train} }
        "dense" { Invoke-Step -Name "dense" -Action ${function:Run-Dense} }
        "rerank" { Invoke-Step -Name "rerank" -Action ${function:Run-Rerank} }
        "all" {
            Invoke-Step -Name "dense" -Action ${function:Run-Dense}
            Invoke-Step -Name "train" -Action ${function:Run-Train}
            Invoke-Step -Name "rerank" -Action ${function:Run-Rerank}
        }
    }

    if ($Benchmark) {
        Write-BenchmarkSummary
    }
}

Write-Host "Completed profile run."
