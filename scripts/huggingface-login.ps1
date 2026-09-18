$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
$form = New-Object Windows.Forms.Form
$form.Text = 'Hugging Face - Local Login'
$form.ClientSize = New-Object Drawing.Size(570, 220)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$label = New-Object Windows.Forms.Label
$label.Text = "Paste the complete hf_ write token below.`r`nIt stays on this PC; do not send it in chat."
$label.Location = New-Object Drawing.Point(20, 20)
$label.Size = New-Object Drawing.Size(530, 45)
$box = New-Object Windows.Forms.TextBox
$box.Location = New-Object Drawing.Point(20, 78)
$box.Size = New-Object Drawing.Size(530, 28)
$box.UseSystemPasswordChar = $true
$box.MaxLength = 1024
$status = New-Object Windows.Forms.Label
$status.Location = New-Object Drawing.Point(20, 120)
$status.Size = New-Object Drawing.Size(530, 40)
$status.Text = 'Ctrl+V to paste. No token is placed in command arguments or logs.'
$button = New-Object Windows.Forms.Button
$button.Text = 'Log in'
$button.Location = New-Object Drawing.Point(430, 169)
$button.Size = New-Object Drawing.Size(120, 30)
$button.Add_Click({
    $tokenText = $box.Text.Trim()
    if($tokenText -notmatch '^hf_[A-Za-z0-9_-]{10,1000}$') {
        $status.Text = 'Invalid format. Copy only the complete hf_ token (no quotes or spaces).'
        return
    }
    $button.Enabled = $false
    $status.Text = 'Checking login with huggingface.co...'
    $form.Refresh()
    $process = $null
    try {
        $startInfo = New-Object Diagnostics.ProcessStartInfo
        $startInfo.FileName = $python
        $startInfo.Arguments = '-E -s -X utf8 scripts/publish_models.py --login-stdin'
        $startInfo.WorkingDirectory = $root
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardInput = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = [Diagnostics.Process]::Start($startInfo)
        $process.StandardInput.WriteLine($tokenText)
        $process.StandardInput.Close()
        $tokenText = $null
        $box.Clear()
        $outputTask = $process.StandardOutput.ReadToEndAsync()
        $errorTask = $process.StandardError.ReadToEndAsync()
        while(!$process.WaitForExit(100)) { [Windows.Forms.Application]::DoEvents() }
        $outputText = $outputTask.GetAwaiter().GetResult()
        $errorText = $errorTask.GetAwaiter().GetResult()
        if($process.ExitCode -eq 0) {
            $status.Text = 'Login saved. You may close this window.'
            [Windows.Forms.MessageBox]::Show($outputText, 'Login successful') | Out-Null
            $form.Close()
        } else {
            $status.Text = 'Login failed. You can paste the token again.'
            [Windows.Forms.MessageBox]::Show($errorText, 'Login failed') | Out-Null
        }
    } catch {
        [Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Login error') | Out-Null
    } finally {
        $tokenText = $null
        if($process) { $process.Dispose() }
        if(!$button.IsDisposed) { $button.Enabled = $true }
    }
})
$form.Controls.AddRange(@($label,$box,$status,$button))
$form.AcceptButton = $button
$form.Add_Shown({$box.Focus()})
[void]$form.ShowDialog()
$box.Clear()
$form.Dispose()
