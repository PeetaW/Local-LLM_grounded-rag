# Research Agent 正式本地 Benchmark 操作手冊（Windows 11）

本文件適用於 PR #10 的 **完整 Research Agent / MCP / Ollama / 七階段 RAG**
驗證階段。開發分支程式已完成離線整合測試後，才執行這些需要
Windows、模型及私人文獻索引的測試。

**不要直接 Merge Draft PR，也不要刪除／重建原始 PDF 索引。**
全套實測結果需保留，但私人樣品與文獻不得上傳到公開 Repo。

## 0. 測試前檢查

確定 PC 原有的 RAG 路徑（以 \`E:\Projects\rag_project\` 為**例**，
不是我已知的實際路徑），且你有原本的 \`llm_env\`。

- 原 RAG checkout: \`E:\Projects\rag_project\`
- 獨立 Agent checkout: \`E:\Projects\rag-research-agent-test\`
- 私人研究檔案: \`E:\PrivateResearch\`
- 原始 PPT 模板：放在私人研究資料夾，絕不可 commit

PowerShell:

\`\`\`powershell
cd "E:\Projects\rag_project"
git status
git fetch origin
git worktree add "..\rag-research-agent-test" "origin/feature/research-agent-adaptive-iteration"
cd "..\rag-research-agent-test"
git rev-parse --short HEAD
\`\`\`

若 worktree 已存在，直接 \`cd\`；不要重複執行 \`git worktree add\`。
檢查 \`git status\`，確認自己沒有手動未儲存的修改。

## 1. 依賴隔離

**不要直接對舊 \`llm_env\` 升級套件。** 若舊環境可用 Conda clone：

\`\`\`powershell
conda create -n research_agent_benchmark --clone llm_env -y
conda activate research_agent_benchmark
python -m pip install -r requirements-research-tools.txt -r requirements-research-agent.txt
python --version
\`\`\`

若只做離線測試，亦可建立全新 Python 3.11 環境：
\`conda create -n research_agent_benchmark python=3.11 -y\`。

新工具採用免費／開源 \`numpy\`, \`scipy\`, \`matplotlib\`,
\`python-pptx\`, MCP SDK v2；沒有必要使用付費 API。

## 2. Offline 功能與安全 Benchmark（無須 GPU）

在 Agent 分支目錄執行：

\`\`\`powershell
python -m unittest discover -s tests -p "test_research_agent*.py" -v
python -m unittest discover -s tests -p "test_spectroscopy.py" -v
python -m unittest discover -s tests -p "test_presentation.py" -v
python -m unittest discover -s tests -p "test_slide_planner.py" -v
python -m unittest discover -s tests -p "test_research_tools_integration.py" -v
python -m unittest discover -s tests -p "test_research_tool_router.py" -v
python -m unittest discover -s tests -p "test_research_cli.py" -v

python -m eval.research_agent_eval --output "agent_offline_benchmark.json"
\`\`\`

預期合成資料 Benchmark 所有案例通過，\`false_successes=0\`。
如有錯誤請保存完整 traceback；絕不可只看平均分數宣布通過。

Mock Benchmark **不能**評估論文推理正確率、Fe3O4/Fe2O3
相定量準確度，也不能代表你的 RTX 3090 真實速度。

## 3. 真實 XRD / FTIR / Raman + PPTX（不需 Ollama）

把 XRD/Raman/FTIR 原始儀器匯出放到私人 workspace，
例如 \`E:\PrivateResearch\Spectroscopy\`；不可上傳 GitHub。

填寫 \`examples/research_tools_manifest.example.json\`：
- \`sample_id\`: 不同**合成樣品** S1、S2
- \`measurement_id\`: 同一樣品不同測點 spot1、spot2
- FTIR \`signal_mode\`：\`transmittance\` 或 \`absorbance\`
- \`min_prominence\`：與儀器訊號同單位，需人工檢查噪音門檻
- 每筆 \`path\` 必須位於 private workspace 內
- PPTX 模板亦必須位於 private workspace 內
- \`.sif\` 是二進位，先用可信軟體匯出文字，不能直接解析

\`\`\`powershell
python -m scripts.run_research_agent --workspace "E:\PrivateResearch" --question "分析兩份 XRD Raman FTIR，並依我的樣板生成可編輯的 PPTX" --manifest "E:\PrivateResearch\Spectroscopy\manifest.json" --template "E:\PrivateResearch\Shuo_Hung_Research_Presentation_Template.pptx" --output-dir "reports" --pptx-output "reports\research_deck.pptx" --mode deep --mcp
\`\`\`

檢查：
- \`agent_run_result.json\` 的 \`stop_reason\` 是否為 \`success\`
- \`reports\analysis.json\` 中是否保留每個 sample_id、measurement_id
- 所有候選峰能否對應原始圖，不將峰值當成礦物定量
- \`reports\research_deck.pptx\` 能否在 PowerPoint 編輯文字，
  封面與內容 Layout、字體、頁碼、圖表是否吻合原模板
- 拒絕把無來源或暫定結論寫成 verified finding

這個測試已包含真正的 MCP stdio client/server；不需 GPU。

## 4. Ollama Planner / Deep Reviewer GPU Smoke Test

在第二個 PowerShell 視窗確認 Ollama：

\`\`\`powershell
nvidia-smi
ollama list
curl.exe "http://localhost:11434/api/tags"
\`\`\`

確認 \`--model\` 名稱確實出現在 \`ollama list\`；不要
自行下載大型模型，若未安裝則先停止並記錄。

在 Agent worktree：

\`\`\`powershell
python -m scripts.run_research_agent --workspace "E:\PrivateResearch" --question "分析兩份 XRD Raman FTIR 並生成 PPT" --manifest "E:\PrivateResearch\Spectroscopy\manifest.json" --template "E:\PrivateResearch\Shuo_Hung_Research_Presentation_Template.pptx" --output-dir "reports" --pptx-output "reports\agent_ollama_deck.pptx" --mode deep --ollama-planner --model "qwen3.5:27b" --mcp
\`\`\`

此時 Planner 會呼叫本機 Ollama JSON 結構化輸出；
Deep reviewer 的質疑是**建議**，不是已驗證的科學事實。
記錄執行時間、VRAM、是否成功完成、是否 fallback，
避免多模型反覆 load/unload。

## 5. 凍結原始七階段 RAG V12.1 Baseline（原工作目錄）

此步驟使用舊原始 checkout 與原 \`llm_env\`，
**不能在沒有私人索引的全新 worktree 直接跑**。

\`\`\`powershell
cd "E:\Projects\rag_project"
conda activate llm_env
nvidia-smi
ollama list
python eval/run_eval.py --run --label v12_1_preagent_smoke --ids Q01,Q09
python eval/run_eval.py --run --label v12_1_preagent_full
\`\`\`

保留 \`eval/results/eval_v12_1_preagent_full.json\`、.md、.log；
每題 correctness、grounding、translation、paper selection
與平均/最慢延遲都要人工審核。通過標準依原版
\`PENDING_TASKS.md\` 的 V12.1 gate，不可擅自改指標來追分。

## 6. Research Agent + 實際七階段 RAG Smoke Test（GPU）

**只在上一階段已通過且原本索引完好時執行。**
再切回 Agent worktree、測試 clone 環境。新 CLI 的
\`--rag-working-dir\` 會在程序中切換 CWD 指向原 checkout，
讓 \`main.py\` 依舊讀取原本的 PDF 索引，無須把私人檔案
複製到 GitHub worktree。此動作仍可能依舊版程式碼初始化
記憶庫或檢查索引，請先備份關鍵索引／metadata。

\`\`\`powershell
cd "E:\Projects\rag-research-agent-test"
conda activate research_agent_benchmark

python -m scripts.run_research_agent --workspace "E:\PrivateResearch" --question "Compare reported boron adsorption mechanisms and limitations across the indexed studies" --mode deep --use-rag --rag-working-dir "E:\Projects\rag_project" --ollama-planner --model "qwen3.5:27b"
\`\`\`

檢查 \`E:\PrivateResearch\agent_run_result.json\`：
- \`planned_tasks\` 是否含 \`rag.query\` + \`evidence.review\`
- \`sources\` 是否有實際檢索到的論文，而非模型自行捏造
- \`status\` 是否成功，若 \`needs_input\` 或 \`revision_limit\`，
  需先排查真實證據與流程，不強迫輸出成功
- \`reported_tokens=0\` 只代表舊 RAG 沒有提供可靠使用量，
  **不是 GPU 完全沒耗用資源**
- 校對引用、數值、科學關係與不確定性；看 GPU 延遲

## 7. 回報與驗收

請提供：
- Windows / Python / Ollama / GPU 版本與執行指令
- 上述離線 Benchmark JSON（不含私人資料）
- V12.1 12 題的彙總 JSON 和評估表（私有內容請遮蔽）
- Agent 本地結果中的 stop_reason、attempts、trace、延遲
- 一張 XRD/FTIR/Raman 原始 vs 新解析的比較圖
- 一張 PPTX 封面與內容頁截圖及你的模板一致性回饋

我們會判斷：是否退化、科學證據是否可信、RAG 正確性與
grounding 是否達標、平均延遲能否接受、是否完成工具安全測試。
**未通過之前不能 Merge PR #10。**

## 8. 可選：啟動獨立本地 HTTP 入口

只在本機使用，並自訂 >=24 字元高強度 token：

\`\`\`powershell
$env:RESEARCH_AGENT_WORKSPACE = "E:\PrivateResearch"
$env:RESEARCH_AGENT_TOKEN = "<請自行建立至少 24 字元的隨機密鑰>"
$env:RESEARCH_AGENT_ENABLE_OLLAMA = "0"
python -m rag.research_agent.http_app
\`\`\`

此入口只綁定 \`127.0.0.1:8765\`，需於
\`X-Research-Agent-Token\` HTTP Header 傳入密鑰。
不要直接對外公開 8765 port；完整登入、速率限制、
CSRF／跨來源驗證及部署硬化不是本原型的承諾。
