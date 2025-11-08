const API = location.origin;
const $ = (id) => document.getElementById(id);

const state = { files: [], k: 4, t: 0.1 };

// --- Stats & documents ---
async function loadStats(){
  const res = await fetch(`${API}/stats`);
  const d = await res.json();
  $("stats").textContent = `${d.total_chunks} chunks indexed • Model: ${d.model}`;
}

async function loadDocs(){
  const res = await fetch(`${API}/documents`);
  const d = await res.json();
  $("files").innerHTML = d.documents.map(x=>`
    <div class="doc-item">
      <div class="doc-icon">📄</div>
      <div class="doc-info">
        <div class="doc-name">${escapeHtml(x.filename)}</div>
        <div class="doc-meta">${x.chunks} chunks</div>
      </div>
    </div>
  `).join("") || `<div class="empty-state">No documents uploaded yet</div>`;
}

// --- Drag & Drop / Pick ---
$("pick").onclick = () => $("fileInput").click();
$("drop").ondragover = (e) => { e.preventDefault(); };
$("drop").ondrop = (e) => {
  e.preventDefault();
  state.files = [...e.dataTransfer.files];
  toast(`${state.files.length} file(s) ready to upload`);
};
$("fileInput").onchange = (e) => {
  state.files = [...e.target.files];
  toast(`${state.files.length} file(s) selected`);
};

// --- Upload ---
$("uploadBtn").onclick = async () => {
  if(!state.files.length) return toast("Please select files first", true);
  const fd = new FormData();
  state.files.forEach(f => fd.append("files", f));
  const res = await fetch(`${API}/upload`, { method:"POST", body: fd });
  const data = await res.json();
  toast("Documents uploaded successfully");
  loadStats();
  loadDocs();
  state.files = [];
};

// --- Reset DB ---
$("resetBtn").onclick = async () => {
  if(!confirm("Are you sure you want to clear the entire index?")) return;
  await fetch(`${API}/reset`, { method:"POST" });
  toast("Index cleared");
  loadStats();
  loadDocs();
  $("chunks").innerHTML = "";
};

// --- Search Preview ---
$("k").oninput = e => {
  state.k = +e.target.value;
  $("kVal").textContent = state.k;
};
$("searchBtn").onclick = runSearch;
$("searchQ").onkeydown = (e) => {
  if(e.key === "Enter") runSearch();
};

async function runSearch(){
  const q = $("searchQ").value.trim();
  if(!q) return;
  const res = await fetch(`${API}/search?q=${encodeURIComponent(q)}&k=${state.k}`);
  const d = await res.json();
  $("chunks").innerHTML = d.results.map(r => `
    <div class="chunk">
      <div class="chunk-meta">
        📄 ${escapeHtml(r.filename)} • Chunk ${r.idx} • Distance: ${r.distance?.toFixed ? r.distance.toFixed(3): r.distance}
      </div>
      <div class="chunk-content">${escapeHtml(r.snippet)}</div>
    </div>
  `).join("") || `<div class="empty-state">No results found</div>`;
}

// --- Chat ---
$("t").oninput = e => {
  state.t = +e.target.value;
  $("tVal").textContent = state.t.toFixed(1);
};
$("askBtn").onclick = ask;
$("q").addEventListener("keydown", e => {
  if(e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});

async function ask(){
  const q = $("q").value.trim();
  if(!q) return;
  pushMsg("user", escapeHtml(q));
  $("q").value = "";
  
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question: q, k: state.k, temperature: state.t })
  });
  const data = await res.json();
  
  // ✅ render Markdown
  let assistantMsg = `<div class="message-bubble">${marked.parse(data.answer)}</div>`;

  if (data.sources?.length){
    assistantMsg += `<div class="sources">` + 
      data.sources.map(s => `<span class="source-chip">${escapeHtml(s)}</span>`).join("") +
      `</div>`;
  }
  pushMsg("assistant", assistantMsg);
}


function pushMsg(role, html){
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.innerHTML = html;
  $("msgs").appendChild(el);
  $("msgs").scrollTop = $("msgs").scrollHeight;
}

function toast(msg, warn = false){
  const icon = warn ? "⚠️" : "✓";
  pushMsg("assistant", `<div class="message-bubble">${icon} ${escapeHtml(msg)}</div>`);
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, m => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[m]));
}

// Init
loadStats();
loadDocs();