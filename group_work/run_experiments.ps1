# TieredKV 实验脚本（在 conda work 环境中运行）
# 用法:
#   .\group_work\run_experiments.ps1 quick       # 快速 ablation benchmark
#   .\group_work\run_experiments.ps1 long        # 长上下文 benchmark
#   .\group_work\run_experiments.ps1 full        # 完整 PPL + benchmark（耗时较长）

param(
    [ValidateSet("quick", "long", "full")]
    [string]$Stage = "quick"
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location (Join-Path $Root "src")

$Py = "conda run -n work --no-capture-output python integrated.py"

switch ($Stage) {
    "quick" {
        Invoke-Expression "$Py --mode all --skip_ppl --gen_len 100"
    }
    "long" {
        Invoke-Expression "$Py --mode all --skip_ppl --long_bench --prefill_tokens 4096 --gen_len 100"
    }
    "full" {
        Invoke-Expression "$Py --mode all --gen_len 200"
    }
}

Write-Host "`nResults -> results/results_integrated.json"
