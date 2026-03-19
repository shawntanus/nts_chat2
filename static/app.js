const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit-btn");
const thinkingEl = document.getElementById("thinking");
const codeEl = document.getElementById("code");
const summaryEl = document.getElementById("summary");
const assumptionsEl = document.getElementById("assumptions");
const notesEl = document.getElementById("notes");
const tableHead = document.querySelector("#results-table thead");
const tableBody = document.querySelector("#results-table tbody");
const answerTitle = document.getElementById("answer-title");
const streamState = document.getElementById("stream-state");
const face = document.getElementById("face");
const faceStatus = document.getElementById("face-status");
const progressFill = document.getElementById("progress-fill");
const runChip = document.getElementById("run-chip");
const runElapsed = document.getElementById("run-elapsed");
const stagePills = Array.from(document.querySelectorAll(".stage-pill"));

const phaseProgress = {
  received: 12,
  thinking: 32,
  coding: 58,
  autotask: 82,
  complete: 100,
  error: 100,
};

let typingQueue = "";
let typingTimer = null;
let elapsedTimer = null;
let startedAt = null;
let lastPhase = "idle";

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const question = chip.textContent.trim();
    questionInput.value = question;
    questionInput.focus();
    if (submitButton.disabled) {
      return;
    }
    runQuestion(question);
  });
});

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function startElapsedClock() {
  clearInterval(elapsedTimer);
  startedAt = Date.now();
  runElapsed.textContent = "00:00";
  elapsedTimer = setInterval(() => {
    runElapsed.textContent = formatElapsed(Date.now() - startedAt);
  }, 250);
}

function stopElapsedClock() {
  clearInterval(elapsedTimer);
}

function setProgress(phase) {
  const progress = phaseProgress[phase] ?? 8;
  progressFill.style.width = `${progress}%`;
}

function updateStages(phase) {
  const ordered = ["received", "thinking", "coding", "autotask", "complete"];
  const activeIndex = ordered.indexOf(phase);

  stagePills.forEach((pill, index) => {
    const pillPhase = pill.dataset.phase;
    pill.classList.remove("active", "complete");

    if (phase === "error") {
      if (pillPhase === lastPhase) {
        pill.classList.add("active");
      }
      return;
    }

    if (index < activeIndex) {
      pill.classList.add("complete");
    } else if (pillPhase === phase) {
      pill.classList.add("active");
    }
  });
}

function ensureTypingLoop() {
  if (typingTimer) {
    return;
  }

  thinkingEl.classList.add("typing");
  typingTimer = setInterval(() => {
    if (!typingQueue.length) {
      clearInterval(typingTimer);
      typingTimer = null;
      thinkingEl.classList.remove("typing");
      return;
    }

    const chunk = typingQueue.slice(0, 2);
    typingQueue = typingQueue.slice(2);
    thinkingEl.textContent += chunk;
    thinkingEl.scrollTop = thinkingEl.scrollHeight;
  }, 18);
}

function queueTypedText(text) {
  if (!text) {
    return;
  }
  typingQueue += text;
  ensureTypingLoop();
}

function flushTyping() {
  if (typingQueue.length) {
    thinkingEl.textContent += typingQueue;
    typingQueue = "";
  }
  thinkingEl.classList.remove("typing");
  clearInterval(typingTimer);
  typingTimer = null;
  thinkingEl.scrollTop = thinkingEl.scrollHeight;
}

function resetUi() {
  stopElapsedClock();
  clearInterval(typingTimer);
  typingTimer = null;
  typingQueue = "";
  lastPhase = "idle";
  thinkingEl.textContent = "";
  thinkingEl.classList.remove("typing");
  codeEl.textContent = "Waiting for generated code...";
  summaryEl.textContent = "Working on your request. Results will appear here as soon as Autotask responds.";
  assumptionsEl.textContent = "";
  notesEl.innerHTML = "";
  tableHead.innerHTML = "";
  tableBody.innerHTML = "";
  answerTitle.textContent = "Answer";
  streamState.textContent = "Running";
  runChip.textContent = "Running";
  faceStatus.textContent = "Thinking through the request";
  face.classList.add("thinking");
  face.classList.remove("success");
  setProgress("received");
  updateStages("received");
  startElapsedClock();
}

function renderTable(columns = [], rows = []) {
  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  const normalized = normalizeTable(columns, rows);
  const safeColumns = normalized.columns;
  const safeRows = normalized.rows;

  if (!safeColumns.length) {
    return;
  }

  const tr = document.createElement("tr");
  safeColumns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    tr.appendChild(th);
  });
  tableHead.appendChild(tr);

  safeRows.forEach((row) => {
    const rowEl = document.createElement("tr");
    row.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value ?? "";
      rowEl.appendChild(td);
    });
    tableBody.appendChild(rowEl);
  });
}

function normalizeTable(columns = [], rows = []) {
  let safeColumns = Array.isArray(columns) ? [...columns] : columns ? [String(columns)] : [];
  let safeRows;

  if (Array.isArray(rows)) {
    safeRows = [...rows];
  } else if (rows && typeof rows === "object") {
    safeRows = [rows];
  } else if (rows !== null && rows !== undefined && rows !== "") {
    safeRows = [[rows]];
  } else {
    safeRows = [];
  }

  const normalizedRows = safeRows.map((row) => {
    if (Array.isArray(row)) {
      return [...row];
    }
    if (row && typeof row === "object") {
      if (!safeColumns.length) {
        safeColumns = Object.keys(row);
      }
      return safeColumns.map((column) => row[column]);
    }
    return [row];
  });

  if (normalizedRows.length && !safeColumns.length) {
    const width = Math.max(...normalizedRows.map((row) => row.length));
    safeColumns = Array.from({ length: width }, (_, index) => `Column ${index + 1}`);
  }

  const paddedRows = normalizedRows.map((row) => {
    if (!safeColumns.length) {
      return row;
    }
    const copy = row.slice(0, safeColumns.length);
    while (copy.length < safeColumns.length) {
      copy.push("");
    }
    return copy;
  });

  return { columns: safeColumns, rows: paddedRows };
}

function appendNotes(notes = []) {
  notesEl.innerHTML = "";
  const safeNotes = Array.isArray(notes)
    ? notes
    : notes && typeof notes === "object"
      ? Object.entries(notes).map(([key, value]) => `${key}: ${value}`)
      : notes !== null && notes !== undefined && notes !== ""
        ? [notes]
        : [];
  safeNotes.forEach((note) => {
    const li = document.createElement("li");
    li.textContent = note;
    notesEl.appendChild(li);
  });
}

function applyStatus(phase, message) {
  lastPhase = phase;
  streamState.textContent = phase;
  runChip.textContent = phase === "autotask" ? "Querying Autotask" : phase;
  faceStatus.textContent = message;
  setProgress(phase);
  updateStages(phase);

  if (phase === "autotask") {
    summaryEl.textContent = "Autotask is responding slowly. The UI will keep showing progress while the query finishes.";
  }
}

async function submitQuestion(question) {
  resetUi();
  submitButton.disabled = true;
  submitButton.textContent = "Running...";

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok || !response.body) {
    throw new Error("The chat request failed.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const event = JSON.parse(line);

      if (event.type === "status") {
        applyStatus(event.phase, event.message);
      }

      if (event.type === "thinking_delta") {
        queueTypedText(event.delta);
      }

      if (event.type === "plan") {
        answerTitle.textContent = event.title || "Answer";
        if (Array.isArray(event.assumptions) && event.assumptions.length) {
          assumptionsEl.textContent = `Assumptions: ${event.assumptions.join(" | ")}`;
        }
        if (Array.isArray(event.reasoning_summary) && event.reasoning_summary.length) {
          queueTypedText(`\n\n${event.reasoning_summary.map((item) => `- ${item}`).join("\n")}`);
        }
      }

      if (event.type === "code") {
        codeEl.textContent = event.code;
      }

      if (event.type === "result") {
        flushTyping();
        face.classList.remove("thinking");
        face.classList.add("success");
        faceStatus.textContent = "Finished with a fresh answer";
        streamState.textContent = "Complete";
        runChip.textContent = "Complete";
        setProgress("complete");
        updateStages("complete");
        summaryEl.textContent = event.result.summary || "Completed.";
        renderTable(event.result.columns || [], event.result.rows || []);
        appendNotes(event.result.notes || []);
      }

      if (event.type === "error") {
        flushTyping();
        face.classList.remove("thinking");
        streamState.textContent = "Error";
        runChip.textContent = "Error";
        faceStatus.textContent = "Hit a snag while processing";
        setProgress("error");
        updateStages("error");
        summaryEl.textContent = event.message;
        if (event.details) {
          notesEl.innerHTML = `<li>${event.details}</li>`;
        }
      }

      if (event.type === "done") {
        stopElapsedClock();
        face.classList.remove("thinking");
        submitButton.disabled = false;
        submitButton.textContent = "Run Question";
      }
    }
  }

  submitButton.disabled = false;
  submitButton.textContent = "Run Question";
}

async function runQuestion(question) {
  if (!question) {
    return;
  }

  try {
    await submitQuestion(question);
  } catch (error) {
    flushTyping();
    stopElapsedClock();
    face.classList.remove("thinking");
    streamState.textContent = "Error";
    runChip.textContent = "Error";
    faceStatus.textContent = "Could not start the request";
    summaryEl.textContent = error.message;
    setProgress("error");
    updateStages("error");
    submitButton.disabled = false;
    submitButton.textContent = "Run Question";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  await runQuestion(question);
});
