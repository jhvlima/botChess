"use strict";

const pieceSymbols = {
  K: "♔",
  Q: "♕",
  R: "♖",
  B: "♗",
  N: "♘",
  P: "♙",
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};

const pieceNames = {
  K: "Rei branco",
  Q: "Dama branca",
  R: "Torre branca",
  B: "Bispo branco",
  N: "Cavalo branco",
  P: "Peão branco",
  k: "Rei preto",
  q: "Dama preta",
  r: "Torre preta",
  b: "Bispo preto",
  n: "Cavalo preto",
  p: "Peão preto",
};

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const elements = {
  board: document.querySelector("#board"),
  status: document.querySelector("#status"),
  color: document.querySelector("#human-color"),
  mode: document.querySelector("#play-mode"),
  newGame: document.querySelector("#new-game"),
  flipBoard: document.querySelector("#flip-board"),
  modelName: document.querySelector("#model-name"),
  botColor: document.querySelector("#bot-color"),
  humanColor: document.querySelector("#human-color-label"),
  lastBotMove: document.querySelector("#last-bot-move"),
  botDecision: document.querySelector("#bot-decision"),
  result: document.querySelector("#result"),
  historyBody: document.querySelector("#history-body"),
  historyScroll: document.querySelector("#history-scroll"),
  promotionDialog: document.querySelector("#promotion-dialog"),
  cancelPromotion: document.querySelector("#cancel-promotion"),
};

let game = null;
let selectedSquare = null;
let pendingPromotionMoves = [];
let busy = false;
let flipped = false;
let errorMessage = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  if (response.status === 204) {
    return null;
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.message || `Falha HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

async function startNewGame() {
  const oldGameId = game?.id;
  busy = true;
  errorMessage = null;
  selectedSquare = null;
  render();
  try {
    const nextGame = await api("/api/games", {
      method: "POST",
      body: JSON.stringify({
        human_color: elements.color.value,
        mode: elements.mode.value,
      }),
    });
    game = nextGame;
    elements.mode.value = game.mode;
    flipped = game.human_color === "black";
    if (oldGameId) {
      await api(`/api/games/${oldGameId}`, { method: "DELETE" }).catch(() => null);
    }
  } catch (error) {
    errorMessage = error.message;
  } finally {
    busy = false;
    render();
  }
}

async function sendMove(uci) {
  if (!game || busy) {
    return;
  }
  busy = true;
  errorMessage = null;
  selectedSquare = null;
  render();
  try {
    game = await api(`/api/games/${game.id}/moves`, {
      method: "POST",
      body: JSON.stringify({ uci }),
    });
  } catch (error) {
    errorMessage = error.message;
  } finally {
    busy = false;
    render();
  }
}

function parseFen(fen) {
  const board = new Map();
  const ranks = fen.split(" ")[0].split("/");
  ranks.forEach((rank, rankIndex) => {
    let fileIndex = 0;
    for (const token of rank) {
      if (/\d/.test(token)) {
        fileIndex += Number(token);
      } else {
        const square = `${files[fileIndex]}${8 - rankIndex}`;
        board.set(square, token);
        fileIndex += 1;
      }
    }
  });
  return board;
}

function squareAriaLabel(square, piece, isTarget) {
  const content = piece ? `${pieceNames[piece]} em ${square}` : `Casa ${square} vazia`;
  return isTarget ? `${content}, destino legal` : content;
}

function renderBoard() {
  const focusedSquare = document.activeElement?.dataset?.square || null;
  elements.board.replaceChildren();
  const position = game ? parseFen(game.fen) : new Map();
  const rankOrder = flipped ? [1, 2, 3, 4, 5, 6, 7, 8] : [8, 7, 6, 5, 4, 3, 2, 1];
  const fileOrder = flipped ? [...files].reverse() : files;
  const legalMoves = game?.legal_moves || [];
  const legalSources = new Set(legalMoves.map((move) => move.slice(0, 2)));
  const selectedTargets = new Set(
    selectedSquare
      ? legalMoves
          .filter((move) => move.slice(0, 2) === selectedSquare)
          .map((move) => move.slice(2, 4))
      : [],
  );
  const lastMoveSquares = new Set(
    game?.last_move ? [game.last_move.slice(0, 2), game.last_move.slice(2, 4)] : [],
  );
  const checkedKing = game?.check
    ? [...position.entries()].find(([, piece]) => piece === (game.turn === "white" ? "K" : "k"))?.[0]
    : null;

  rankOrder.forEach((rank, rowIndex) => {
    fileOrder.forEach((file, columnIndex) => {
      const square = `${file}${rank}`;
      const piece = position.get(square);
      const isLight = (files.indexOf(file) + rank) % 2 === 0;
      const isTarget = selectedTargets.has(square);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `square ${isLight ? "light" : "dark"}`;
      button.dataset.square = square;
      button.setAttribute("aria-label", squareAriaLabel(square, piece, isTarget));
      button.setAttribute("aria-pressed", selectedSquare === square ? "true" : "false");
      button.tabIndex = focusedSquare
        ? (focusedSquare === square ? 0 : -1)
        : (rowIndex === 0 && columnIndex === 0 ? 0 : -1);

      if (legalSources.has(square)) button.classList.add("movable");
      if (lastMoveSquares.has(square)) button.classList.add("last-move");
      if (selectedSquare === square) button.classList.add("selected");
      if (isTarget) button.classList.add("legal-target");
      if (isTarget && piece) button.classList.add("capture");
      if (checkedKing === square) button.classList.add("in-check");

      if (piece) {
        const pieceElement = document.createElement("span");
        pieceElement.className = `piece ${piece === piece.toUpperCase() ? "white-piece" : "black-piece"}`;
        pieceElement.textContent = pieceSymbols[piece];
        button.append(pieceElement);
      }
      if (rowIndex === 7) {
        const fileLabel = document.createElement("span");
        fileLabel.className = "coordinate file-label";
        fileLabel.textContent = file;
        button.append(fileLabel);
      }
      if (columnIndex === 0) {
        const rankLabel = document.createElement("span");
        rankLabel.className = "coordinate rank-label";
        rankLabel.textContent = String(rank);
        button.append(rankLabel);
      }
      button.addEventListener("click", () => selectSquare(square));
      button.addEventListener("keydown", moveBoardFocus);
      elements.board.append(button);
    });
  });
  if (focusedSquare) {
    elements.board
      .querySelector(`[data-square="${focusedSquare}"]`)
      ?.focus({ preventScroll: true });
  }
}

function moveBoardFocus(event) {
  const directions = {
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
  };
  const direction = directions[event.key];
  if (!direction) return;
  const squares = [...elements.board.querySelectorAll(".square")];
  const currentIndex = squares.indexOf(event.currentTarget);
  const row = Math.floor(currentIndex / 8) + direction[0];
  const column = (currentIndex % 8) + direction[1];
  if (row < 0 || row > 7 || column < 0 || column > 7) return;
  event.preventDefault();
  const target = squares[row * 8 + column];
  event.currentTarget.tabIndex = -1;
  target.tabIndex = 0;
  target.focus();
}

function selectSquare(square) {
  if (!game || busy || game.game_over || game.turn !== game.human_color) {
    return;
  }
  const legalMoves = game.legal_moves || [];
  const movesFrom = (source) => legalMoves.filter((move) => move.slice(0, 2) === source);
  errorMessage = null;

  if (selectedSquare) {
    if (selectedSquare === square) {
      selectedSquare = null;
    } else {
      const candidates = movesFrom(selectedSquare).filter(
        (move) => move.slice(2, 4) === square,
      );
      if (candidates.length === 1) {
        sendMove(candidates[0]);
        return;
      }
      if (candidates.length > 1) {
        requestPromotion(candidates);
        return;
      }
      selectedSquare = movesFrom(square).length ? square : null;
    }
  } else if (movesFrom(square).length) {
    selectedSquare = square;
  }
  render();
}

function requestPromotion(candidates) {
  pendingPromotionMoves = candidates;
  const promotionSymbols = game?.human_color === "black"
    ? { q: "♛", r: "♜", b: "♝", n: "♞" }
    : { q: "♕", r: "♖", b: "♗", n: "♘" };
  for (const button of document.querySelectorAll("[data-promotion]")) {
    button.textContent = promotionSymbols[button.dataset.promotion];
  }
  if (typeof elements.promotionDialog.showModal === "function") {
    elements.promotionDialog.showModal();
    return;
  }
  const choice = window.prompt("Promoção: q (dama), r (torre), b (bispo) ou n (cavalo)", "q");
  const move = candidates.find((candidate) => candidate.endsWith(choice?.toLowerCase()));
  if (move) sendMove(move);
}

function renderHistory() {
  elements.historyBody.replaceChildren();
  if (!game?.history?.length) {
    const row = document.createElement("tr");
    row.className = "empty-history";
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "A partida ainda não começou.";
    row.append(cell);
    elements.historyBody.append(row);
    return;
  }

  const grouped = new Map();
  for (const move of game.history) {
    const row = grouped.get(move.move_number) || { white: "", black: "" };
    row[move.color] = move.san;
    grouped.set(move.move_number, row);
  }
  for (const [moveNumber, moves] of grouped) {
    const row = document.createElement("tr");
    for (const value of [moveNumber, moves.white, moves.black]) {
      const cell = document.createElement("td");
      cell.textContent = value || "—";
      row.append(cell);
    }
    elements.historyBody.append(row);
  }
  elements.historyScroll.scrollTop = elements.historyScroll.scrollHeight;
}

function renderDetails() {
  const colorLabel = (color) => (color === "white" ? "Brancas" : "Pretas");
  const modeLabel = game?.mode === "coherent" ? "Coerente" : "Imitador";
  elements.modelName.textContent = game
    ? `${game.model} · ${modeLabel}`
    : "carregando modelo";
  elements.humanColor.textContent = game ? colorLabel(game.human_color) : "—";
  elements.botColor.textContent = game ? colorLabel(game.bot_color) : "—";
  elements.result.textContent = game?.result || "";
  elements.lastBotMove.textContent = game?.last_bot_move
    ? `${game.last_bot_move.san} · ${game.last_bot_move.uci}`
    : "Aguardando o primeiro lance";

  const decisionReasons = {
    mate_in_one: "encontrou um mate imediato",
    avoided_mate: "evitou permitir mate imediato",
    avoided_stalemate: "evitou afogamento",
    material_guard: "evitou uma perda material grave",
  };
  const decision = game?.last_bot_move;
  const showDecision = Boolean(
    decision?.coherence_applied && decisionReasons[decision.selection_reason],
  );
  elements.botDecision.hidden = !showDecision;
  elements.botDecision.textContent = showDecision
    ? `A proteção coerente trocou a 1ª opção pela ${decision.style_rank}ª: `
      + `${decisionReasons[decision.selection_reason]}.`
    : "";
}

function renderStatus() {
  elements.status.classList.toggle("error", Boolean(errorMessage));
  elements.status.classList.toggle("finished", Boolean(game?.game_over && !errorMessage));
  if (errorMessage) {
    elements.status.textContent = errorMessage;
  } else if (busy) {
    elements.status.textContent = "O bot está pensando…";
  } else if (game) {
    elements.status.textContent = game.message;
  } else {
    elements.status.textContent = "Preparando a partida…";
  }
}

function render() {
  elements.newGame.disabled = busy;
  elements.color.disabled = busy;
  elements.mode.disabled = busy;
  elements.flipBoard.disabled = busy || !game;
  renderStatus();
  renderBoard();
  renderDetails();
  renderHistory();
}

elements.newGame.addEventListener("click", startNewGame);
elements.flipBoard.addEventListener("click", () => {
  flipped = !flipped;
  renderBoard();
});
elements.cancelPromotion.addEventListener("click", () => {
  pendingPromotionMoves = [];
  elements.promotionDialog.close();
});
elements.promotionDialog.addEventListener("cancel", () => {
  pendingPromotionMoves = [];
});
for (const button of document.querySelectorAll("[data-promotion]")) {
  button.addEventListener("click", () => {
    const move = pendingPromotionMoves.find((candidate) =>
      candidate.endsWith(button.dataset.promotion),
    );
    pendingPromotionMoves = [];
    elements.promotionDialog.close();
    if (move) sendMove(move);
  });
}

render();
startNewGame();
