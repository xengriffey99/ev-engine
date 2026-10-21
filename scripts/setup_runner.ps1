<#
One-time setup of the Phase 0 snapshot runner on Windows (PowerShell 5.1+).

  powershell -ExecutionPolicy Bypass -File scripts\setup_runner.ps1            # Maine (default)
  powershell -ExecutionPolicy Bypass -File scripts\setup_runner.ps1 -State MA  # Massachusetts

- Stores the Odds API key encrypted to your Windows account (DPAPI) in
  %APPDATA%\nba-ev\odds_api_key.dpapi. It never appears in a task definition.
- Registers five scheduled tasks under \NBA-EV\ with "wake the computer to run",
  so a sleeping PC still takes the T-5 close.
- Verifies the key with the free /events call (0 credits).
UNTESTED on Windows by Claude (no PowerShell in the build sandbox). Run it once and send the output if anything fails.
#>
param(
    [ValidateSet("ME", "MA")] [string]$State = "ME",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$ConfDir = Join-Path $env:APPDATA "nba-ev"
$KeyFile = Join-Path $ConfDir "odds_api_key.dpapi"
New-Item -ItemType Directory -Force -Path $ConfDir | Out-Null

# 1. Python 3.11+
& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required on PATH (or pass -Python <path>)." }

# 2. Key (DPAPI-encrypted to this user)
if (-not (Test-Path $KeyFile)) {
    $secure = Read-Host "Paste your Odds API key (input hidden)" -AsSecureString
    $secure | ConvertFrom-SecureString | Set-Content -Path $KeyFile
    Write-Host "Key stored encrypted at $KeyFile"
} else {
    Write-Host "Key already stored at $KeyFile (delete the file to replace it)."
}
Set-Content -Path (Join-Path $ConfDir "bet_state") -Value $State

# 3. Wrapper the tasks call
$Wrapper = Join-Path $Repo "scripts\nba_ev.ps1"
@'
param([Parameter(Mandatory=$true)][string]$Cmd)
$Repo = Split-Path -Parent $PSScriptRoot
$ConfDir = Join-Path $env:APPDATA "nba-ev"
Set-Location $Repo
New-Item -ItemType Directory -Force -Path "data\logs" | Out-Null
$et = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, "Eastern Standard Time")
$Log = "data\logs\runner_{0}.log" -f $et.ToString("yyyy-MM")
$secure = Get-Content (Join-Path $ConfDir "odds_api_key.dpapi") | ConvertTo-SecureString
$env:ODDS_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
$env:NBA_EV_BET_STATE = (Get-Content (Join-Path $ConfDir "bet_state") -ErrorAction SilentlyContinue) -as [string]
if (-not $env:NBA_EV_BET_STATE) { $env:NBA_EV_BET_STATE = "ME" }
$env:PYTHONPATH = Join-Path $Repo "src"
$args2 = @($Cmd)
if ($Cmd -eq "opinions-today") {
    $payload = "data\payload_{0}.json" -f $et.ToString("yyyy-MM-dd")
    if (-not (Test-Path $payload)) { Add-Content $Log "$(Get-Date -Format o) no model payload at $payload"; exit 0 }
    $args2 = @("opinions", "--payload", $payload)
}
Add-Content $Log "=== $(Get-Date -Format o) $($args2 -join ' ') (state=$env:NBA_EV_BET_STATE)"
& __PYTHON__ -m market.run_snapshots @args2 *>> $Log
'@.Replace("__PYTHON__", $Python) | Set-Content -Path $Wrapper

# 4. Scheduled tasks (times are ET; the PC must be on Eastern time)
if ((Get-TimeZone).Id -ne "Eastern Standard Time") {
    Write-Warning "This PC is not on Eastern time; the schedule assumes ET."
}
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable:$false -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
function Add-NbaTask($Name, $Cmd, $Trigger) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Wrapper`" -Cmd $Cmd" -WorkingDirectory $Repo
    Register-ScheduledTask -TaskPath "\NBA-EV\" -TaskName $Name -Action $action -Trigger $Trigger `
        -Settings $settings -Force | Out-Null
    Write-Host "Registered \NBA-EV\$Name"
}
$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Add-NbaTask "run-due"        "run-due"        $every5
Add-NbaTask "plan"           "plan"           (New-ScheduledTaskTrigger -Daily -At "09:00")
Add-NbaTask "opinions-today" "opinions-today" (New-ScheduledTaskTrigger -Daily -At "12:15")
Add-NbaTask "close-out"      "close-out"      (New-ScheduledTaskTrigger -Daily -At "02:30")
Add-NbaTask "export"         "export"         (New-ScheduledTaskTrigger -Daily -At "02:45")

# 5. Verify with the free /events call
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Wrapper -Cmd plan
Get-ChildItem (Join-Path $Repo "data\logs") | Sort-Object LastWriteTime | Select-Object -Last 1 |
    ForEach-Object { Get-Content $_.FullName -Tail 3 }
Write-Host "Done. Under Power Options, enable 'Allow wake timers' so the PC wakes for the T-5 closes."
