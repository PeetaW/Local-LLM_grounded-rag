# Windows 本地測試指南：Research Agent / XRD–FTIR–Raman / PPTX

> 目的：保護舊版 RAG baseline，使用獨立 Git worktree 測試新工具。
> 本指南只在使用者準備於自己的 Windows 11 / Ollama / NVIDIA GPU 執行
> 完整端到端測試時使用；此前先以 GitHub Actions 離線測試。
>
> **勿將私人實驗原始數據、原始個人簡報模板或實驗室內部資料提交到公開 GitHub。**

## A. 在 Windows PowerShell 準備獨立分支工作目錄

以你現在本地的 Repo 根目錄為例（請換成實際路徑）：

\`\`\`powershell
cd "E:\Projects\rag_project"
git status
git fetch origin
git worktree add "..\rag-research-agent-test" "origin/feature/research-agent-adaptive-iteration"
cd "..\rag-research-agent-test"
git rev-parse --short HEAD
\`\`\`

此 worktree 只用於測試，不切換原本 \`main\` 或舊工作目錄；
若 worktree 已存在，直接進入該資料夾即可，不要再次執行 add。

## B. 啟動 Python 環境、安裝可選工具依賴

\`\`\`powershell
conda activate llm_env
python --version
python -m pip install -r requirements-research-tools.txt
\`\`\`

只安裝免費、開源的 \`numpy\`、\`scipy\`、\`matplotlib\`、
\`python-pptx\`。若擔心現有環境版本衝突，請改在新 Conda
測試環境安裝（\`conda create -n research_tools_test python=3.11\`）。

## C. 不使用 Ollama 的測試

\`\`\`powershell
python -m unittest discover -s tests -p "test_research_agent.py" -v
python -m unittest discover -s tests -p "test_spectroscopy.py" -v
python -m unittest discover -s tests -p "test_presentation.py" -v
python -m unittest discover -s tests -p "test_slide_planner.py" -v
python -m unittest discover -s tests -p "test_research_tools_integration.py" -v
python -m unittest discover -s tests -p "test_research_cli.py" -v
\`\`\`

全部測試應正常完成，不能忽略 Fail/Error。
若 GitHub Actions 已顯示全部通過，這一步只是對 Windows
相容性進行確認。

## D. 使用你自己的 XRD、Raman、FTIR 原始檔測試

1. 建立**不在公開 Repo 內**的資料夾，例如 \`E:\PrivateResearch\Spectroscopy\`，
   放入儀器匯出的 XRD \`.txt\`、Raman \`.asc\`、
   FTIR \`.csv\`；保留原始文件不改寫。
2. 複製 \`examples/research_tools_manifest.example.json\` 至
   \`E:\PrivateResearch\Spectroscopy\manifest.json\`。
3. 打開 JSON，修改每筆 \`path\` 為自己的真實資料路徑。
   在 Windows JSON 建議用正斜線，例如：
   \`"path": "E:/PrivateResearch/Spectroscopy/20261102 Fe3O4_1.txt"\`。
4. \`sample_id\` 對應不同實驗樣品，例如 S1、S2。
   同一樣品的不同測點使用不同 \`measurement_id\`（spot1、spot2）。
   不可將不同測點假裝成獨立合成批次。
5. FTIR 必須指定 \`signal_mode\` 為 \`transmittance\`（%T）
   或 \`absorbance\`，勿憑檔名猜測。
6. \`min_prominence\` 是**原始 Y 軸單位**下的候選峰強度門檻，
   範例數字只是教學起點，必須檢查雜訊與峰偵測品質。
7. 把你私有的
   \`Shuo_Hung_Research_Presentation_Template.pptx\`
   放在安全路徑。別 commit 模板。

\`\`\`powershell
python -m scripts.run_research_tools_smoke --manifest "E:\PrivateResearch\Spectroscopy\manifest.json" --output "E:\PrivateResearch\Spectroscopy\out" --template "E:\PrivateResearch\Shuo_Hung_Research_Presentation_Template.pptx" --title "Fe3O4 XRD Raman FTIR"
\`\`\`

輸出內容：
- \`out\analysis.json\`：原始檔來源、樣品 ID、各測點候選峰與限制。
- \`out\spectrum_001.png\` 等：各測量圖表（FTIR 左側為高波數）。
- \`out\research_report.pptx\`：可編輯文字與頁面、圖表為 PNG，
  保留模板主題與適當的封面/內容 Layout。

**目前尚不能**從候選峰直接判定
Fe₃O₄/γ-Fe₂O₃/α-Fe₂O₃ 的可靠 wt.%、純度或定量比例。
請核對峰位置、光譜雜訊、樣品測點區分以及模板視覺位置，
若有疑問保留原始數據進行人工比對。

## E. 下一道 GPU/V12.1 Baseline（僅在通知後執行）

需使用原本的 RAG 專案論文索引、Ollama 模型與實際 NVIDIA GPU。
**不要在本階段任意調整** \`config.py\`、
embedding/reranker/top-k/num_ctx 或重新建索引。

先於另一個 PowerShell 視窗檢查：

\`\`\`powershell
nvidia-smi
ollama list
curl.exe "http://localhost:11434/api/tags"
\`\`\`

先執行選定題目確認環境；若沒有錯誤，再執行全 12 題：

\`\`\`powershell
python eval/run_eval.py --run --label v12_1_preagent_smoke --ids Q01,Q09
python eval/run_eval.py --run --label v12_1_preagent_full
\`\`\`

\`--run\` 會使用真實 Ollama 模型；測試時保留原本
\`ACTIVE_PROJECT\` 和其 PDF 索引，避免不同 corpus 造成
不公平比較。測試結束後保留：

- \`eval/results/eval_v12_1_preagent_full.json\`
- \`eval/results/eval_v12_1_preagent_full.md\`
- \`eval/results/eval_v12_1_preagent_full.log\`
- 你當時的 \`nvidia-smi\` GPU 型號/VRAM/使用量截圖（如有）

若本地已有 \`baseline_v12_candidate_full\` 的完整評估檔，
可再執行：

\`\`\`powershell
python eval/run_eval.py --compare baseline_v12_candidate_full v12_1_preagent_full
\`\`\`

若舊評估檔不存在，跳過比較指令即可。不要將包含私人文獻、
絕對路徑或內部數據的 log 上傳到公開 PR。

## F. 回報資訊（錯誤或品質問題）

提供：執行指令、最後 30–50 行錯誤訊息、
CI Run/Commit SHA、\`analysis.json\` 中非機密欄位、
一至兩張輸出光譜圖、PPTX 截圖、GPU 與 Ollama 模型版本。

V12.1 原始 quality gate 規則以 \`PENDING_TASKS.md\` 為準；
研究工具的獨立單元測試不代表完成原始七階段模型品質驗收。
