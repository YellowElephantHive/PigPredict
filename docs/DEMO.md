# Demo walkthrough / 展示步驟

## A two-minute walkthrough

1. Open the studio. The first measured sample automatically produces candidate recipes. Introduce the task: **given a target color and fabric, find suitable dye combinations and concentrations**.
2. Point to the paired target and model-predicted swatches. Explain that the network predicts Lab from a recipe, and the inverse solver adjusts concentrations toward the target.
3. Switch from **Recipe 01** to **Recipe 03**. Show the different dye combination and the change in predicted ΔE₇₆.
4. Return to Recipe 01. Change the first dye concentration from approximately **0.0640** to **0.5000**. The predicted color, Lab differences and total concentration update immediately.
5. Export the edited recipe. The CSV retains both target and predicted Lab values, making the result usable outside the screen.
6. Load **Coral** or **Blue iris** to show another target and fabric. Switch to **中文** when presenting to a Taiwanese audience.
7. Open **Model details & engineering** to discuss model conversion, browser/Python consistency, deterministic optimization and the Django interface.

Reloading the page always restores English. Recipe results are computed by the original trained decoder; they are not prewritten result cards.

## 中文講法

「這個專案解決的是染整配色問題。給定目標顏色與布料，我們要找出可以使用的染料組合與濃度。我的做法是先讓神經網路學會配方和 Lab 顏色之間的關係，再固定模型權重，反向搜尋接近目標色的濃度。」

「這裡可以比較多組候選配方。如果我改變其中一種染料的濃度，右側預測色與色差會立即改變。最後可以匯出配方，保留目標色、預測色和實際調整的濃度。」

「原始版本使用 Django 與 TensorFlow。這次重新整理展示介面時，我保留模型權重，補上瀏覽器推論與數值一致性測試，讓作品可以直接透過網址操作。」

## English explanation

“This project translates a target textile color into candidate dye recipes. The neural network learns the forward relationship from dye concentrations and fabric type to Lab color. Inverse search then adjusts the concentrations while keeping the trained model fixed.”

“The interface makes that relationship interactive. I can compare dye combinations, change a concentration, inspect the predicted color difference, and export the resulting recipe.”

“The original application used Django and TensorFlow. The refreshed version preserves the trained weights, adds browser inference, and verifies its numerical consistency against the original model artifacts.”
