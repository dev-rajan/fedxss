/**
 * background.js — service worker: owns the model, scores incoming records.
 *
 * The model lives here rather than in the content script so it is parsed
 * once per browser session instead of once per page. Nothing leaves the
 * device: there is no network code in this extension at all, which is the
 * property Section 3.2 claims and the one a reviewer can verify by reading
 * this file.
 *
 * Detections are held in memory only and are lost when the worker unloads.
 * Persisting a log of flagged page content would recreate exactly the
 * centralised record of browsing behaviour the study sets out to avoid.
 */

import { featurise } from "./features.js";
import { Detector } from "./model.js";

let detector = null;
let loading = null;

const stats = {
  scored: 0,
  flagged: 0,
  totalScoreMs: 0,
  recent: [],            // capped at 50; in-memory only, never persisted
};

async function ensureLoaded() {
  if (detector) return detector;
  if (!loading) {
    loading = (async () => {
      const url = chrome.runtime.getURL("model.json");
      const payload = await (await fetch(url)).json();
      detector = new Detector(payload);
      console.log("[fedxss] model loaded:", payload.provenance);
      return detector;
    })();
  }
  return loading;
}

async function scoreRecords(records) {
  const det = await ensureLoaded();
  const results = [];
  for (const r of records) {
    const t0 = performance.now();
    const out = det.predict(featurise(r));
    const dt = performance.now() - t0;

    stats.scored++;
    stats.totalScoreMs += dt;
    if (out.flagged) {
      stats.flagged++;
      stats.recent.unshift({
        url: (r.url || "").slice(0, 200),
        snippet: (r.script || "").slice(0, 160),
        sink: r.sink,
        probability: out.probability,
        at: Date.now(),
      });
      stats.recent.length = Math.min(stats.recent.length, 50);
    }
    results.push({ ...out, ms: dt });
  }
  return results;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "score") {
    scoreRecords(msg.records)
      .then((results) => {
        const n = results.filter((r) => r.flagged).length;
        if (n > 0 && sender.tab?.id != null) {
          chrome.action.setBadgeText({ tabId: sender.tab.id, text: String(n) });
          chrome.action.setBadgeBackgroundColor({ tabId: sender.tab.id, color: "#b00020" });
        }
        sendResponse({ ok: true, results });
      })
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;   // async response
  }
  if (msg?.type === "stats") {
    ensureLoaded().then((d) => sendResponse({
      ok: true,
      stats: {
        ...stats,
        meanScoreMs: stats.scored ? stats.totalScoreMs / stats.scored : 0,
      },
      provenance: d.provenance,
    }));
    return true;
  }
});
