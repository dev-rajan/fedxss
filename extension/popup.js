chrome.runtime.sendMessage({ type: "stats" }, (resp) => {
  const prov = document.getElementById("prov");
  // Surface failures instead of leaving the popup on "loading model…".
  if (chrome.runtime.lastError) {
    prov.textContent = "extension error: " + chrome.runtime.lastError.message;
    return;
  }
  if (!resp?.ok) {
    prov.textContent = "model failed to load" + (resp?.error ? ": " + resp.error : "");
    return;
  }
  const s = resp.stats, p = resp.provenance || {};
  prov.textContent =
    `${p.training || "model"} · test F1 ${Number(p.test_f1 || 0).toFixed(4)}`;
  document.getElementById("scored").textContent = s.scored;
  document.getElementById("flagged").textContent = s.flagged;
  document.getElementById("ms").textContent = `${s.meanScoreMs.toFixed(3)} ms`;

  if (!s.recent.length) return;
  const box = document.getElementById("recent");
  box.innerHTML = "";
  for (const r of s.recent.slice(0, 12)) {
    const d = document.createElement("div");
    d.className = "item";
    const pct = document.createElement("span");
    pct.className = "p";
    pct.textContent = `${(r.probability * 100).toFixed(1)}%  `;
    const c = document.createElement("code");
    // textContent, never innerHTML: this popup renders attacker-controlled
    // markup, and an XSS detector that is itself XSS-vulnerable would be a
    // poor advertisement for the work.
    c.textContent = r.snippet || r.url;
    d.append(pct, c);
    box.append(d);
  }
});
