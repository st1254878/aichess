// dark_chess_board.js
const setupDiv = document.getElementById("setup");
const gameDiv = document.getElementById("game");
const boardElement = document.getElementById("board");
const turnIndicator = document.getElementById("turnIndicator");
const thinkingEl = document.getElementById("thinking");

let selectedStart = null;
let selectedEnd = null;
let humanFirst = true;
let currentSessionId = null;

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

        // render 初始棋盤
        await updateBoard();

    } catch (err) {
        console.error("startGame error:", err);
        alert("建立遊戲發生錯誤，請看 console");
    }
}

// ------------------ 建立棋盤 ------------------
function createBoard() {
    boardElement.innerHTML = "";
    for (let r = 0; r < 4; r++) {
        const row = document.createElement("div");
        row.className = "row flex";
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement("div");
            cell.className = "cell border border-gray-400 w-16 h-16 flex items-center justify-center text-xl font-bold";
            cell.style.border = "1px solid gray";
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.addEventListener("click", onCellClick);
            row.appendChild(cell);
        }
        boardElement.appendChild(row);
    }
}

// ------------------ 點擊格子 ------------------
function onCellClick(event) {
    const cell = event.currentTarget; // use currentTarget to ensure div not img
    const row = cell.dataset.row;
    const col = cell.dataset.col;

    // Toggle selection logic: first click 設 start，再次同一格取消；第二次設 end
    if (!selectedStart) {
        selectedStart = { row, col, cell };
        cell.style.border = "4px solid orange";
    } else if (!selectedEnd) {
        selectedEnd = { row, col, cell };
        cell.style.border = "4px solid red";
    } else {
        // 已經選了 start 和 end，再點會重設成新的 start
        // 清掉之前的選擇樣式
        resetSelection();
        selectedStart = { row, col, cell };
        cell.style.border = "4px solid orange";
    }
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

// ------------------ 重置選擇 ------------------
function resetSelection() {
    selectedStart = null;
    selectedEnd = null;
    document.querySelectorAll(".cell").forEach(cell => {
        cell.style.border = "1px solid gray";
    });
}

function resetGame() {
  // 清除棋盤與狀態
  boardElement.innerHTML = "";
  turnIndicator.innerText = "";
  thinkingEl.style.display = "none";

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
