# Credit-risk modeling pipeline - PowerShell entry point.
#
#   .\run.ps1            # full pipeline
#   .\run.ps1 pd         # a single stage (data|quality|sql|eda|pd|lgd|clustering|simulation)
#   .\run.ps1 test       # unit tests
#
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Stages = @("all"))

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ($Stages.Count -eq 1 -and $Stages[0] -eq "test") {
        python -m pytest -q
    }
    elseif ($Stages.Count -eq 1 -and $Stages[0] -eq "install") {
        python -m pip install -r requirements.txt
    }
    else {
        python -m src.pipeline @Stages
    }
}
finally {
    Pop-Location
}
