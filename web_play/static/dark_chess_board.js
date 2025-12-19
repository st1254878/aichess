// dark_chess_board.js
const setupDiv = document.getElementById("setup");
const gameDiv = document.getElementById("game");
const boardElement = document.getElementById("board");
const turnIndicator = document.getElementById("turnIndicator");
const thinkingEl = document.getElementById("thinking");
const thinkingEl2 = document.getElementById("thinking2");
let humanColor = null;
let selectedStart = null;
let selectedEnd = null;
let humanFirst = true;
let firstmove = true;
let currentSessionId = null;
let isSubmitting = false;
let interactionDisabled = false;
// 棋子對應的圖片路徑
const pieceImages = {
    "暗棋": "static/imgs/blankchess.png",
    "红车": "static/imgs/hongche.png",
    "红马": "static/imgs/hongma.png",
    "红象": "static/imgs/hongxiang.png",
    "红士": "static/imgs/hongshi.png",
    "红帅": "static/imgs/hongshuai.png",
    "红炮": "static/imgs/hongpao.png",
    "红兵": "static/imgs/hongbing.png",
    "黑车": "static/imgs/heiche.png",
    "黑马": "static/imgs/heima.png",
    "黑象": "static/imgs/heixiang.png",
    "黑士": "static/imgs/heishi.png",
    "黑帅": "static/imgs/heishuai.png",
    "黑炮": "static/imgs/heipao.png",
    "黑兵": "static/imgs/heibing.png",
};

// ------------------ Helpers ------------------
function setSessionId(sid) {
    currentSessionId = sid;
    try { sessionStorage.setItem('aichess_session_id', sid); } catch (e) {}
}
function getSessionId() {
    if (currentSessionId) return currentSessionId;
    try {
        return sessionStorage.getItem('aichess_session_id');
    } catch (e) {
        return null;
    }
}


// ------------------ 開始遊戲 ------------------
async function startGame(side) {
    thinkingEl2.style.display = 'block'
    try {
        const res = await fetch("/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ side })
        });
        const data = await res.json();
        if (!res.ok) {
            alert("建立遊戲失敗：" + (data.error || JSON.stringify(data)));
            return;
        }

        // 儲存 session id
        setSessionId(data.session_id);

        humanFirst = data.human_first;

        setupDiv.style.display = "none";
        gameDiv.style.display = "block";
        createBoard();
        if (data.human_color) {
            humanColor = data.human_color;
        } else {
            humanColor = null;
        }
        // render 初始棋盤
        await updateBoard();

    } catch (err) {
        console.error("startGame error:", err);
        alert("建立遊戲發生錯誤，請看 console");
    }
    thinkingEl2.style.display = 'none';
}

// ------------------ 建立棋盤 ------------------
function createBoard() {
    boardElement.innerHTML = "";
    for (let r = 0; r < 4; r++) {
        const row = document.createElement("div");
        row.className = "row"; // grid 已處理格子排版，row 可留空或用於語義
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement("div");
            cell.className = "cell";
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.addEventListener('touchend', onCellTouchEnd, { passive: false });
            cell.addEventListener('click', onCellTouchEnd, { passive: false }); // 桌機仍支援滑鼠
            row.appendChild(cell);
        }
        boardElement.appendChild(row);
    }
}

function updateTurnIndicator(currentPlayer) {
    if (firstmove) {
        firstmove = false;
        if (humanFirst) {
            return;
        }
    }
    let badgeHtml = '';
    if (humanColor === '红') badgeHtml = '<span class="player-badge badge-red" title="你是紅方"></span>';
    else if (humanColor === '黑') badgeHtml = '<span class="player-badge badge-black" title="你是黑方"></span>';

    let text = '';
    if (humanColor) {
        if (currentPlayer) {
            if (currentPlayer === humanColor) text = `${badgeHtml} 現在輪到你 (${currentPlayer})`;
            else text = `${badgeHtml} 現在輪到暗棋阿拉法 (${currentPlayer})`;
        }
    } else {
        text = currentPlayer ? `目前輪到：${currentPlayer}` : '目前輪到：-';
    }
    turnIndicator.innerHTML = text;
}

// ------------------ 點擊格子（touchend / click handler） ------------------
function onCellTouchEnd(event) {
    // 阻止 touchend 後觸發 click
    try { event.preventDefault(); } catch (e) {}

    if (interactionDisabled) return; // 禁止互動時忽略點擊

    const cell = event.currentTarget;
    const row = cell.dataset.row;
    const col = cell.dataset.col;

    // 若沒有起點 -> 設為起點
    if (!selectedStart) {
        selectedStart = { row, col, cell };
        cell.classList.add('start-selected');
        return;
    }

    // 已有起點但沒有終點
    if (selectedStart && !selectedEnd) {
        // 同格：視為翻棋（也算一個 move）
        if (selectedStart.row === row && selectedStart.col === col) {
            selectedEnd = { row, col, cell };
            cell.classList.add('end-selected');
            // 立即嘗試送出 move（翻棋）
            attemptMove();
            return;
        }

        // 不同格：視為移動，立即送出
        selectedEnd = { row, col, cell };
        cell.classList.add('end-selected');
        attemptMove();
        return;
    }

    // 已有 start & end -> 重新當作新的 start
    resetSelection();
    selectedStart = { row, col, cell };
    cell.classList.add('start-selected');
}

// ------------------ 嘗試送出走步（取代 confirmMove） ------------------
async function attemptMove() {
    if (isSubmitting) return; // 已在送出中
    const sid = getSessionId();
    if (!sid) {
        alert("找不到 session，請先按 Start 開新遊戲");
        resetSelection();
        return;
    }
    if (!selectedStart || !selectedEnd) {
        // 保險：若沒有完整選擇就忽略
        return;
    }

    isSubmitting = true;
    disableInteraction();

    const move_action = `${selectedStart.row}${selectedStart.col}${selectedEnd.row}${selectedEnd.col}`;

    try {
        const res = await fetch("/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sid, move_action })
        });
        const data = await res.json();

        if (!res.ok) {
            // 伺服器回錯誤，例如 illegal move
            const msg = data.error || JSON.stringify(data);
            // 若是 illegal move，可給更友善說明
            if (data.error === 'illegal move') {
                alert("走法不合法");
            } else if (data.error === 'game_already_over') {
                alert("遊戲已結束");
            } else {
                alert("錯誤：" + msg);
            }
            // 失敗時：把終點取消（保留起點），讓玩家繼續選終點或取消
            if (selectedEnd && selectedEnd.cell) {
                resetSelection();
            }
            return;
        }

        // 成功：伺服器可能回傳 board, game_over, 等資訊
        // 更新畫面
        await updateBoard();

        // 如果人方這步導致遊戲結束，就顯示結果並不叫 AI
        if (data.game_over) {
            if (data.winner) {
                alert(data.winner === 1 ? "紅方勝利！" : (data.winner === 2 ? "黑方勝利！" : "平局"));
            } else {
                alert("遊戲結束！");
            }
            // 清除選擇
            resetSelection();
            return;
        }

        // 若遊戲未結且是人下完，要讓 AI 行動
        await aiMove();

        // AI 做完之後也會更新棋盤（aiMove 中已呼 updateBoard）
        // 可以選擇清除選擇或保留（建議清除）
        resetSelection();

    } catch (err) {
        console.error("attemptMove error:", err);
        alert("送出走步時發生錯誤，請看 console");
        // 若失敗，重置選擇以免卡狀態
        resetSelection();
    } finally {
        isSubmitting = false;
        enableInteraction();
    }
}

// 簡單禁用/啟用交互（如果你想要在送出期間不讓玩家再點）
function disableInteraction() {
    interactionDisabled = true;
    // optional: show visual feedback
    thinkingEl.style.display = "block";
}
function enableInteraction() {
    interactionDisabled = false;
    thinkingEl.style.display = "none";
}

// ------------------ 確認走棋 ------------------
async function confirmMove() {
    if (!selectedStart || !selectedEnd) {
        alert("請先選擇起點與終點");
        return;
    }
    const sid = getSessionId();
    if (!sid) {
        alert("找不到 session，請先按「先手/後手」建立新遊戲");
        return;
    }

    const move_action = `${selectedStart.row}${selectedStart.col}${selectedEnd.row}${selectedEnd.col}`;

    try {
        const res = await fetch("/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sid, move_action })
        });
        const data = await res.json();
        if (!res.ok) {
            alert(`錯誤：${data.error || JSON.stringify(data)}`);
            resetSelection();
            return;
        }

        // 更新畫面
        await updateBoard();

        // 讓 AI 走（如果遊戲還沒結束）
        await aiMove();

    } catch (err) {
        console.error("confirmMove error:", err);
        alert("送出走步時發生錯誤，請看 console");
    } finally {
        resetSelection();
    }
}

// ------------------ 更新棋盤 ------------------
async function updateBoard() {
    const sid = getSessionId();
    if (!sid) {
        console.warn("updateBoard: missing session");
        return;
    }
    try {
        const res = await fetch(`/state?session_id=${encodeURIComponent(sid)}`);
        const data = await res.json();
        if (!res.ok) {
            console.error("state error:", data);
            alert("取得棋盤狀態錯誤：" + (data.error || JSON.stringify(data)));
            return;
        }
        const board = data.board;
        const currentPlayer = data.current_player;


        const cells = document.querySelectorAll(".cell");
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 8; c++) {
                const idx = r * 8 + c;
                const cell = cells[idx];
                updateCell(cell, board[r][c]);
            }
        }

        updateTurnIndicator(currentPlayer);

    } catch (err) {
        console.error("updateBoard error:", err);
    }
}

function updateCell(cell, piece) {
    cell.innerHTML = ""; // 清空格子內容
    cell.style.background = ""; // reset background if any

    // some servers might return null or "一一" for empty
    if (piece && piece !== "一一") {
        const img = document.createElement("img");
        img.src = pieceImages[piece] || pieceImages["暗棋"];
        img.alt = piece;
        img.className = "piece-image";
        img.style.maxWidth = "90%";
        img.style.maxHeight = "90%";
        cell.appendChild(img);
    }
}

// ------------------ AI 走棋 ------------------
async function aiMove() {
    const sid = getSessionId();
    if (!sid) {
        alert("找不到 session，請先按 Start 開新遊戲");
        return;
    }

    thinkingEl.style.display = "block";

    try {
        const res = await fetch("/ai_move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sid })
        });
        const data = await res.json();
        if (!res.ok) {
            console.error("ai_move failed", data);
            alert("AI 執行時發生錯誤：" + (data.error || JSON.stringify(data)));
            return;
        }
        // 更新棋盤
        await updateBoard();

    } catch (err) {
        console.error("AI move error:", err);
        alert("AI 執行時發生錯誤（請看 console）");
    } finally {
        thinkingEl.style.display = "none";
    }
}

function resetSelection() {
    selectedStart = null;
    selectedEnd = null;
    document.querySelectorAll(".cell").forEach(cell => {
        cell.classList.remove('start-selected', 'end-selected');
    });
}

function resetGame() {
  // 清除棋盤與狀態
  boardElement.innerHTML = "";
  turnIndicator.innerText = "";
  thinkingEl.style.display = "none";
  firstmove = true;
  // 清除 session id
  currentSessionId = null;
  try { localStorage.removeItem('aichess_session_id'); } catch (e) {}

  // 顯示選擇先後手畫面，隱藏遊戲畫面
  setupDiv.style.display = "block";
  gameDiv.style.display = "none";
}

// ------------------ 頁面載入時自動恢復 session（若存在） ------------------
window.addEventListener('load', async () => {
  const sid = getSessionId();
  if (sid) {
    // 嘗試取得狀態，若成功就直接進遊戲畫面
    try {
      const res = await fetch(`/state?session_id=${encodeURIComponent(sid)}`);
      if (res.ok) {
        const data = await res.json();
        // 恢復畫面
        setupDiv.style.display = "none";
        gameDiv.style.display = "block";
        createBoard();
        await updateBoard();
        return;
      }
    } catch (e) {
      console.warn("Restore session failed:", e);
    }
    // 若失敗，清掉已儲存的 session id
    try { localStorage.removeItem('aichess_session_id'); } catch (e) {}
  }
});
