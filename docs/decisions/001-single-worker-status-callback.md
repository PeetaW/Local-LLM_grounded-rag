# ADR 001：Pipeline 進度訊息的串流設計

## 狀態
已實作（V3）

## 決策
使用 status_callback（on_status 參數）模式傳遞 pipeline 進度訊息，
而非攔截全域 sys.stdout。

## 背景
V2 使用 _StatusCapture 攔截 sys.stdout，搭配 max_workers=1 迴避 race condition。
此設計可運作，但假設是隱性的，且未來難以擴展。

_StatusCapture 的問題：
- 修改全域 sys.stdout，多 thread 並發時會有 race condition
- 難以測試（必須 monkey-patch sys.stdout）
- 不同請求的進度訊息無法區分

## 取捨
- callback 模式：每個 request 有獨立的 handler，天然並發安全，可測試性高
- stdout 攔截：實作簡單，但依賴全域狀態，並發不安全

## 硬體限制說明
ThreadPoolExecutor max_workers=1 是硬體限制，不是設計限制。
RTX 3090（24GB VRAM）搭配 gemma4:31b（26GB）無法真正並發推理。
改為 status_callback 模式後，若未來多卡部署可直接調整 max_workers 數值。

## 觸發重新評估的條件
若未來需要把進度訊息寫入 log 系統或 metrics，
在 status_handler 內加入對應邏輯即可，不需修改 pipeline 本身。
