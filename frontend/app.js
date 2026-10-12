"use strict";

// Credentials and chat content deliberately never enter browser storage.
const ui = Object.fromEntries(Array.from(document.querySelectorAll("[id]"), (el) => [el.id, el]));
const state = { token: "", course: "default", session: null, controller: null, busy: false, connecting: false, loadingSession: false, uploading: false, generation: 0, audioUrls: new Set() };
const welcome = ui.welcome.cloneNode(true);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function notice(message = "") {
  ui.notice.textContent = message;
  ui.notice.hidden = !message;
}

function safeApiUrl(path) {
  const url = new URL(path, window.location.origin);
  const allowedPath = url.pathname.startsWith("/api/") || /^\/audio\/[0-9a-f]{32}\.wav$/.test(url.pathname);
  if (url.origin !== window.location.origin || !allowedPath) {
    throw new Error("The server returned an invalid API URL.");
  }
  return url.href;
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers);
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const controller = options.signal ? null : new AbortController();
  const timer = controller ? setTimeout(() => controller.abort(), 45000) : null;
  let response;
  try {
    response = await fetch(safeApiUrl(path), { ...options, headers, signal: options.signal || controller.signal, cache: "no-store", credentials: "same-origin" });
  } catch (error) {
    if (controller?.signal.aborted) throw new Error("The request timed out. Check the connection and try again.");
    throw error;
  } finally { if (timer) clearTimeout(timer); }
  if (!response.ok) {
    let detail = "";
    try { const body = await response.json(); detail = typeof body.detail === "string" ? body.detail : typeof body.message === "string" ? body.message : ""; } catch { /* Non-JSON gateway responses use the status below. */ }
    const fallback = response.status === 401 ? "Access denied. Check your classroom token and reconnect." : response.status === 403 ? "Your token does not have permission for this course or operation." : response.status === 429 ? "The classroom is busy. Please wait a moment and try again." : `Request failed (${response.status}). Please try again.`;
    throw new Error(detail || fallback);
  }
  return response;
}

async function json(path, options) {
  const response = await request(path, options);
  return response.status === 204 ? null : response.json();
}

function updateControls() {
  const locked = state.busy || state.connecting || state.loadingSession;
  ui["send-btn"].disabled = locked;
  ui["send-btn"].hidden = state.busy;
  ui["stop-btn"].hidden = !state.busy;
  for (const id of ["connect-btn", "course-id", "api-token"]) ui[id].disabled = locked || state.uploading;
  for (const id of ["new-chat", "refresh-sessions", "refresh-documents"]) ui[id].disabled = locked;
  ui["upload-btn"].disabled = locked || state.uploading;
  ui["document-file"].disabled = locked || state.uploading;
  ui["voice-select"].disabled = !ui["with-audio"].checked || state.busy;
  ui["with-audio"].disabled = state.busy;
  ui["with-rag"].disabled = state.busy;
  for (const button of ui["session-list"].querySelectorAll("button")) button.disabled = locked;
}

function syncConnection() {
  const course = ui["course-id"].value.trim();
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(course)) throw new Error("Use a course ID of 1–64 letters, numbers, hyphens or underscores, starting with a letter or number.");
  const token = ui["api-token"].value.trim();
  if (token.length < 32) throw new Error("Enter your classroom access token (at least 32 characters) to connect.");
  if (course !== state.course || token !== state.token) {
    state.course = course;
    state.token = token;
    state.generation += 1;
    clearConversation();
    ui["session-list"].replaceChildren(element("p", "empty-list", "Connect to load conversations."));
    ui["document-list"].replaceChildren();
    ui["upload-status"].textContent = "";
    ui["job-list"].replaceChildren();
    ui["connection-status"].textContent = "Connection changed";
    ui["connection-status"].classList.remove("connected");
  }
}

function clearConversation() {
  state.session = null;
  for (const url of state.audioUrls) URL.revokeObjectURL(url);
  state.audioUrls.clear();
  ui.messages.replaceChildren(welcome.cloneNode(true));
  ui["conversation-title"].textContent = "Classroom assistant";
  ui["stream-status"].textContent = "";
  markSelected();
}

function markSelected() {
  for (const row of ui["session-list"].querySelectorAll(".session-row")) {
    const selected = row.dataset.id === state.session;
    row.classList.toggle("selected", selected);
    row.querySelector(".session-open").setAttribute("aria-current", selected ? "true" : "false");
  }
}

function scrollMessages(force = false) {
  const node = ui.messages;
  if (force || node.scrollHeight - node.scrollTop - node.clientHeight < 160) node.scrollTop = node.scrollHeight;
}

function addMessage(role, text = "") {
  ui.messages.querySelector(".welcome")?.remove();
  const article = element("article", `message ${role === "user" ? "user" : "assistant"}`);
  article.append(element("div", "message-label", role === "user" ? "You" : "AI Teacher"));
  const content = element("div", "message-content", text);
  article.append(content);
  ui.messages.append(article);
  scrollMessages(true);
  return { article, content };
}

function renderSources(article, sources) {
  article.querySelector(".source-section")?.remove();
  if (!Array.isArray(sources) || !sources.length) return;
  const details = element("details", "source-section");
  details.append(element("summary", "", `${sources.length} course source${sources.length === 1 ? "" : "s"}`));
  const grid = element("div", "source-grid");
  sources.forEach((source, index) => {
    const card = element("section", "source-card");
    const page = source.page != null ? ` · p. ${source.page}` : "";
    card.append(element("h3", "", `[${source.id || index + 1}] ${source.title || "Course document"}${page}`));
    card.append(element("p", "", source.text || ""));
    grid.append(card);
  });
  details.append(grid);
  article.append(details);
}

function attachAudio(article, path) {
  const container = element("div", "audio-container");
  const button = element("button", "audio-load", "▶ Listen to explanation");
  button.type = "button";
  container.append(button);
  article.append(container);
  const generation = state.generation;
  button.addEventListener("click", async () => {
    if (generation !== state.generation) return;
    button.disabled = true;
    button.textContent = "Loading audio…";
    try {
      const response = await request(path);
      const blob = await response.blob();
      if (!article.isConnected || generation !== state.generation) return;
      const url = URL.createObjectURL(blob);
      state.audioUrls.add(url);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = url;
      audio.setAttribute("aria-label", "Spoken assistant response");
      container.replaceChildren(audio);
      audio.focus();
    } catch (error) {
      button.disabled = false;
      button.textContent = "Retry loading audio";
      if (generation === state.generation) notice(error.message);
    }
  });
}

async function refreshSessions() {
  const generation = state.generation;
  const data = await json(`/api/sessions?course_id=${encodeURIComponent(state.course)}`);
  if (generation !== state.generation) return;
  const sessions = Array.isArray(data) ? data : data.sessions || data.items || [];
  ui["session-list"].replaceChildren();
  if (!sessions.length) ui["session-list"].append(element("p", "empty-list", "Your next question starts a conversation."));
  sessions.forEach((session) => {
    const id = String(session.id || session.session_id);
    const title = session.title || "Untitled conversation";
    const row = element("div", "session-row");
    row.dataset.id = id;
    const open = element("button", "session-open", title);
    open.type = "button";
    open.title = title;
    open.addEventListener("click", () => openSession(id));
    const remove = element("button", "delete-button", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Delete conversation: ${title}`);
    remove.addEventListener("click", async () => {
      if (state.busy || state.loadingSession || !window.confirm(`Delete “${title}” and its messages?`)) return;
      remove.disabled = true;
      try {
        await request(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (state.session === id) clearConversation();
        await refreshSessions();
      } catch (error) { notice(error.message); remove.disabled = false; }
    });
    row.append(open, remove);
    ui["session-list"].append(row);
  });
  markSelected();
  updateControls();
}

async function openSession(id) {
  if (state.busy || state.loadingSession) return;
  state.loadingSession = true;
  updateControls();
  notice();
  try {
    syncConnection();
    const data = await json(`/api/sessions/${encodeURIComponent(id)}`);
    clearConversation();
    state.session = id;
    ui["conversation-title"].textContent = data.title || "Classroom conversation";
    for (const message of data.messages || []) {
      if (!["user", "assistant"].includes(message.role)) continue;
      const { article } = addMessage(message.role, message.content);
      renderSources(article, message.sources);
      if (message.audio_url) attachAudio(article, message.audio_url);
      if (Array.isArray(message.warnings) && message.warnings.length) article.append(element("p", "message-note", message.warnings.join(" ")));
    }
    markSelected();
    ui.prompt.focus();
  } catch (error) { notice(error.message); }
  finally { state.loadingSession = false; updateControls(); }
}

function documentIdentity(doc) {
  return String(doc.document_id || doc.id || "");
}

async function refreshDocuments() {
  const generation = state.generation;
  const course = state.course;
  const data = await json(`/api/documents?course_id=${encodeURIComponent(course)}`);
  if (generation !== state.generation) return;
  const docs = Array.isArray(data) ? data : data.documents || data.items || [];
  ui["document-list"].replaceChildren();
  if (!docs.length) ui["document-list"].append(element("li", "empty-list", "No course documents indexed yet."));
  docs.forEach((doc) => {
    const id = documentIdentity(doc);
    const row = element("li", "document-row");
    const label = element("div", "document-label", doc.title || doc.filename || id);
    const status = doc.status || "Indexed";
    const chunks = doc.chunks ?? doc.chunk_count;
    const count = chunks != null ? ` · ${chunks} chunks` : "";
    label.append(element("small", "", `${status}${count}`));
    const remove = element("button", "delete-button", "×");
    remove.type = "button";
    remove.disabled = !id;
    remove.setAttribute("aria-label", `Delete document: ${doc.title || doc.filename || id}`);
    remove.addEventListener("click", async () => {
      if (generation !== state.generation || !window.confirm("Remove this document from the course knowledge base?")) return;
      remove.disabled = true;
      try { await request(`/api/documents/${encodeURIComponent(id)}?course_id=${encodeURIComponent(course)}`, { method: "DELETE" }); await refreshDocuments(); }
      catch (error) { notice(error.message); remove.disabled = false; }
    });
    row.append(label, remove);
    ui["document-list"].append(row);
  });
}

async function monitorJob(id, generation, label, statusNode) {
  try {
    // Bounded polling: indexing remains a durable server job if this tab closes.
    for (let attempt = 0; attempt < 180; attempt += 1) {
      if (generation !== state.generation) return;
      const job = await json(`/api/jobs/${encodeURIComponent(id)}`);
      if (generation !== state.generation) return;
      const status = String(job.status || "pending").toLowerCase();
      statusNode.textContent = `${label} · ${status}`;
      if (["done", "completed", "succeeded", "ready"].includes(status)) {
        statusNode.textContent = `${label} · indexed and ready for questions`;
        await refreshDocuments();
        return;
      }
      if (["failed", "error", "cancelled"].includes(status)) throw new Error(typeof job.error === "string" ? job.error : "Document indexing failed. Check the document and try again.");
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    statusNode.textContent = `${label} · still indexing; refresh documents later to check the result`;
  } catch (error) {
    if (generation === state.generation) { statusNode.textContent = `${label} · ${error.message}`; }
  }
}

// Parse SSE incrementally: UTF-8 boundaries, CRLF and multi-line data are supported.
async function readEvents(response, onEvent) {
  if (!response.body) throw new Error("Your browser cannot read streaming responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data = [];
  function line(value) {
    if (value === "") {
      if (data.length) {
        const payload = data.join("\n");
        try { onEvent(event, JSON.parse(payload)); }
        catch (error) { if (error instanceof SyntaxError) throw new Error("The server sent an unreadable stream event."); throw error; }
      }
      event = "message";
      data = [];
    } else if (value.startsWith("event:")) event = value.slice(6).trim();
    else if (value.startsWith("data:")) data.push(value.slice(5).replace(/^ /, ""));
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let index;
      while ((index = buffer.indexOf("\n")) !== -1) {
        line(buffer.slice(0, index).replace(/\r$/, ""));
        buffer = buffer.slice(index + 1);
      }
      if (done) { if (buffer) line(buffer.replace(/\r$/, "")); line(""); break; }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.busy || state.connecting || state.loadingSession) return;
  const prompt = ui.prompt.value.trim();
  if (!prompt) return;
  try { syncConnection(); } catch (error) { notice(error.message); return; }
  state.busy = true;
  state.controller = new AbortController();
  updateControls();
  notice();
  addMessage("user", prompt);
  ui.prompt.value = "";
  ui.prompt.style.height = "";
  const { article, content } = addMessage("assistant");
  article.classList.add("pending");
  ui["stream-status"].textContent = ui["with-rag"].checked ? "Finding relevant course material…" : "Preparing your explanation…";
  let completed = false;
  let received = false;
  try {
    const response = await request("/api/chat/stream", {
      method: "POST", signal: state.controller.signal,
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ prompt, course_id: state.course, session_id: state.session, with_audio: ui["with-audio"].checked, voice: ui["voice-select"].value, rag: ui["with-rag"].checked }),
    });
    if (!response.headers.get("content-type")?.includes("text/event-stream")) throw new Error("The server did not return a supported event stream.");
    await readEvents(response, (type, data) => {
      if (type === "meta") {
        if (data.session_id) state.session = data.session_id;
        ui["stream-status"].textContent = "Preparing your explanation…";
      } else if (type === "sources") renderSources(article, data.sources);
      else if (type === "token") {
        received = true;
        content.textContent += data.text || "";
        ui["stream-status"].textContent = "Writing your explanation…";
        scrollMessages();
      } else if (type === "done") {
        completed = true;
        if (typeof data.text === "string") content.textContent = data.text;
        if (data.session_id) state.session = data.session_id;
        if (data.sources) renderSources(article, data.sources);
        if (data.audio_url) attachAudio(article, data.audio_url);
        if (Array.isArray(data.warnings) && data.warnings.length) article.append(element("p", "message-note", data.warnings.join(" ")));
        ui["stream-status"].textContent = "Response complete.";
      } else if (type === "error") throw new Error(data.message || "The response could not be completed.");
    });
    if (!completed) throw new Error("The connection ended before the answer was complete. You can retry your question.");
  } catch (error) {
    const stopped = error.name === "AbortError";
    const message = stopped ? "Generation stopped. This answer may be incomplete." : error.message;
    article.append(element("p", "message-note", message));
    ui["stream-status"].textContent = stopped ? "Generation stopped." : "Response interrupted.";
    if (!stopped) notice(error.message);
    if (!received && !completed) { content.textContent = "No answer received."; if (!ui.prompt.value) ui.prompt.value = prompt; }
  } finally {
    article.classList.remove("pending");
    state.busy = false;
    state.controller = null;
    updateControls();
    markSelected();
    ui.prompt.focus();
    refreshSessions().catch(() => { /* Preserve the answer when history refresh fails. */ });
  }
}

ui["chat-form"].addEventListener("submit", sendMessage);
ui["stop-btn"].addEventListener("click", () => state.controller?.abort());
ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && window.matchMedia("(min-width: 801px)").matches) {
    event.preventDefault();
    ui["chat-form"].requestSubmit();
  }
});
ui.prompt.addEventListener("input", () => { ui.prompt.style.height = "auto"; ui.prompt.style.height = `${Math.min(ui.prompt.scrollHeight, 180)}px`; });
ui["with-audio"].addEventListener("change", updateControls);
ui["with-rag"].addEventListener("change", () => { ui["grounding-badge"].textContent = ui["with-rag"].checked ? "✧ Course-grounded" : "✧ General tutoring"; });
for (const id of ["course-id", "api-token"]) {
  ui[id].addEventListener("input", () => {
    // Invalidate immediately; outstanding requests cannot render another identity's data.
    state.generation += 1;
    state.token = "";
    clearConversation();
    ui["session-list"].replaceChildren(element("p", "empty-list", "Reconnect to load conversations."));
    ui["document-list"].replaceChildren();
    ui["upload-status"].textContent = "";
    ui["job-list"].replaceChildren();
    ui["connection-status"].textContent = "Not connected";
    ui["connection-status"].classList.remove("connected");
  });
}
ui["new-chat"].addEventListener("click", () => { if (!state.busy) { clearConversation(); notice(); ui.prompt.focus(); } });
ui.messages.addEventListener("click", (event) => {
  const button = event.target.closest(".suggestion");
  if (!button) return;
  ui.prompt.value = button.dataset.prompt;
  ui.prompt.focus();
});
ui["connection-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || state.connecting || state.uploading) return;
  notice();
  try { syncConnection(); } catch (error) { notice(error.message); return; }
  state.connecting = true;
  updateControls();
  ui["connection-status"].textContent = "Connecting…";
  try {
    await refreshSessions();
    ui["connection-status"].textContent = `Connected · ${state.course}`;
    ui["connection-status"].classList.add("connected");
    if (ui["knowledge-panel"].open) await refreshDocuments();
  } catch (error) {
    ui["connection-status"].textContent = "Connection failed";
    ui["connection-status"].classList.remove("connected");
    notice(error.message);
  } finally { state.connecting = false; updateControls(); }
});
ui["refresh-sessions"].addEventListener("click", async () => { try { syncConnection(); await refreshSessions(); } catch (error) { notice(error.message); } });
ui["refresh-documents"].addEventListener("click", async () => { try { syncConnection(); await refreshDocuments(); } catch (error) { notice(error.message); } });
ui["knowledge-panel"].addEventListener("toggle", async () => {
  if (!ui["knowledge-panel"].open) return;
  try { syncConnection(); await refreshDocuments(); } catch (error) { notice(error.message); }
});
ui["upload-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.uploading || state.busy || state.connecting) return;
  const file = ui["document-file"].files[0];
  if (!file) return;
  try { syncConnection(); } catch (error) { notice(error.message); return; }
  const form = new FormData();
  form.append("file", file);
  state.uploading = true;
  updateControls();
  notice();
  ui["upload-status"].textContent = "Uploading document…";
  try {
    const job = await json(`/api/documents?course_id=${encodeURIComponent(state.course)}`, { method: "POST", body: form });
    const id = job.id || job.job_id;
    ui["upload-form"].reset();
    if (id) {
      ui["upload-status"].textContent = "Document queued for indexing.";
      const statusNode = element("li", "", `${file.name} · queued`);
      ui["job-list"].prepend(statusNode);
      void monitorJob(id, state.generation, file.name, statusNode);
    }
    else { ui["upload-status"].textContent = "Document accepted."; await refreshDocuments(); }
  } catch (error) { ui["upload-status"].textContent = "Upload failed."; notice(error.message); }
  finally { state.uploading = false; updateControls(); }
});
function applyTheme(dark) {
  document.body.classList.toggle("dark", dark);
  ui["theme-toggle"].setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} theme`);
}
applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
ui["theme-toggle"].addEventListener("click", () => applyTheme(!document.body.classList.contains("dark")));
window.addEventListener("pagehide", () => { state.controller?.abort(); for (const url of state.audioUrls) URL.revokeObjectURL(url); });
updateControls();
