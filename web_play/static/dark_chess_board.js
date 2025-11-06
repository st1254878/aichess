const setupDiv = document.getElementById("setup");
const gameDiv = document.getElementById("game");
const boardElement = document.getElementById("board");
const turnIndicator = document.getElementById("turnIndicator");

let selectedStart = null;
let selectedEnd = null;
let humanFirst = true;

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

// ------------------ 開始遊戲 ------------------
async function startGame(side) {
    const res = await fetch("/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ side })
    });
    const data = await res.json();
    humanFirst = data.human_first;

    setupDiv.style.display = "none";
    gameDiv.style.display = "block";
    createBoard();
    await updateBoard();
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
    const cell = event.target;
    const row = cell.dataset.row;
    const col = cell.dataset.col;

    if (!selectedStart) {
        selectedStart = { row, col };
        cell.style.border = "4px solid orange";
    } else if (!selectedEnd) {
        selectedEnd = { row, col };
        cell.style.border = "4px solid red";
    }
}

// ------------------ 確認走棋 ------------------
async function confirmMove() {
    if (!selectedStart || !selectedEnd) {
        alert("請先選擇起點與終點");
        return;
    }
    const move_action = `${selectedStart.row}${selectedStart.col}${selectedEnd.row}${selectedEnd.col}`;
    if (!confirm(`確認移動 ${move_action} 嗎？`)) {
        resetSelection();
        return;
    }

    const res = await fetch("/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move_action })
    });
    const data = await res.json();

    if (data.error) {
        alert(`錯誤：${data.error}`);
    } else {
        await updateBoard();
        await aiMove();
    }

    resetSelection();
}

// ------------------ 更新棋盤 ------------------
async function updateBoard() {
    const res = await fetch("/state");
    const data = await res.json();
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
}

function updateCell(cell, piece) {
    cell.innerHTML = ""; // 清空格子內容

    if (piece && piece !== "一一") {
        const img = document.createElement("img");
        img.src = pieceImages[piece] || pieceImages["暗棋"];
        img.alt = piece;
        img.className = "piece-image";
        cell.appendChild(img);
    }
}


// ------------------ AI 走棋 ------------------
async function aiMove() {
    const thinking = document.getElementById("thinking");

    // 顯示「AI 正在思考中...」
    thinking.style.display = "block";

    try {
        const res = await fetch("/ai_move", { method: "POST" });
        const data = await res.json();
        console.log("AI move:", data.ai_move);

        // 更新棋盤
        await updateBoard();

    } catch (err) {
        console.error("AI move error:", err);
        alert("AI 執行時發生錯誤");
    } finally {
        // 無論成功或失敗都隱藏提示
        thinking.style.display = "none";
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
  const board = document.getElementById("board");
  board.innerHTML = "";
  document.getElementById("turnIndicator").innerText = "";

  // 顯示選擇先後手畫面，隱藏遊戲畫面
  document.getElementById("setup").style.display = "block";
  document.getElementById("game").style.display = "none";
}
