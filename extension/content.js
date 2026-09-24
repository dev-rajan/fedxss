/**
 * content.js — collects observation records from the page.
 *
 * Mirrors the three event sources named in Section 3.2: the navigation
 * itself, inline script execution, and DOM mutations that introduce markup.
 * Records are shaped exactly like the training schema (url, script, sink,
 * origin) so the same feature extractor applies unchanged.
 *
 * Batched and debounced, and capped per page: a site that rewrites its DOM
 * continuously would otherwise generate thousands of messages. This is a
 * monitoring prototype, not an auditing tool — missing the ten-thousandth
 * mutation on a page matters less than staying out of the way.
 */

const MAX_RECORDS_PER_PAGE = 200;
const FLUSH_MS = 250;

let queue = [];
let sent = 0;
let timer = null;

function originOf(el) {
  const src = el?.src;
  if (!src) return "same";
  try {
    return new URL(src, location.href).origin === location.origin ? "same" : "cross";
  } catch (e) {
    return "same";
  }
}

function push(record) {
  if (sent >= MAX_RECORDS_PER_PAGE) return;
  queue.push(record);
  sent++;
  if (!timer) timer = setTimeout(flush, FLUSH_MS);
}

function flush() {
  timer = null;
  if (!queue.length) return;
  const records = queue;
  queue = [];
  try {
    chrome.runtime.sendMessage({ type: "score", records }, () => void chrome.runtime.lastError);
  } catch (e) {
    /* extension context invalidated on reload; ignore */
  }
}

// 1. the navigation itself — URL-carried payloads (reflected, DOM-based)
push({ url: location.href, script: "", sink: "none", origin: "same" });

// 2. inline scripts present at load
for (const el of document.querySelectorAll("script")) {
  const text = el.textContent || "";
  if (text.trim()) {
    push({ url: location.href, script: text.slice(0, 600), sink: "eval", origin: "same" });
  } else if (el.src) {
    push({ url: el.src, script: "", sink: "src", origin: originOf(el) });
  }
}

// 3. DOM mutations that introduce markup — the stored and DOM-based paths
const observer = new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      const html = node.outerHTML || "";
      if (!html) continue;
      const tag = node.tagName?.toLowerCase();
      push({
        url: location.href,
        script: html.slice(0, 600),
        sink: tag === "script" ? "eval" : "innerHTML",
        origin: originOf(node),
      });
    }
    if (m.type === "attributes" && m.attributeName?.startsWith("on")) {
      push({
        url: location.href,
        script: `${m.attributeName}=${m.target.getAttribute(m.attributeName) || ""}`,
        sink: "setAttribute",
        origin: "same",
      });
    }
  }
});

observer.observe(document.documentElement, {
  childList: true, subtree: true, attributes: true,
});

window.addEventListener("pagehide", () => { observer.disconnect(); flush(); });
