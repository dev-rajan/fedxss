/**
 * features.js — JavaScript port of src/features.py
 *
 * This file MUST produce vectors identical to the Python extractor. If it
 * drifts, the extension silently runs a different detector from the one
 * evaluated in Chapter 4, and every reported number becomes a claim about
 * something that is not deployed.
 *
 * Parity is checked by tools/parity_check.js against vectors dumped from
 * Python; run it after any change to either file.
 *
 * Two portability decisions made this feasible:
 *   - the extractor is stateless (no fitted vocabulary to ship or version)
 *   - character n-grams use FNV-1a, which is integer arithmetic and behaves
 *     identically in both languages given the same 32-bit masking
 */

const N_HASH = 274;
const OFFSET_HASH = 46;
const DIM = OFFSET_HASH + N_HASH; // 320

const SINKS = [
  "none", "innerHTML", "outerHTML", "document.write", "insertAdjacentHTML",
  "eval", "setAttribute", "src", "href", "textContent",
];

const DANGEROUS_TAGS = /<\s*\/?\s*(script|img|svg|iframe|body|input|details|video|audio|object|embed|marquee|math|form|style|link|meta|base|applet|isindex)\b/gi;
const EVENT_HANDLER = /\bon[a-z]{3,15}\s*=/gi;
const JS_URI = /javascript\s*:/i;
const SINK_CALL = /(innerHTML|outerHTML|document\s*\.\s*write|insertAdjacentHTML|\beval\s*\(|setAttribute|\bFunction\s*\()/i;
const SOURCE_READ = /(location\s*\.\s*(hash|search|href)|window\s*\.\s*name|URLSearchParams|document\s*\.\s*(referrer|cookie|URL))/i;
const ALERTISH = /\b(alert|prompt|confirm|atob|fromCharCode)\s*\(/i;

function countMatches(re, s) {
  const r = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g");
  let n = 0;
  while (r.exec(s) !== null) n++;
  return n;
}

function shannon(s) {
  if (!s) return 0.0;
  const counts = new Map();
  for (const ch of s) counts.set(ch, (counts.get(ch) || 0) + 1);
  const n = s.length;
  let h = 0.0;
  for (const c of counts.values()) {
    const p = c / n;
    h -= p * Math.log2(p);
  }
  return h;
}

// Python's str.isdigit / isalnum / isupper / isspace are Unicode-aware;
// these mirror them closely enough for the ASCII-dominant inputs here.
const isDigit = (ch) => /\p{Nd}/u.test(ch);
const isAlnum = (ch) => /[\p{L}\p{N}]/u.test(ch);
const isUpper = (ch) => /\p{Lu}/u.test(ch);
const isSpace = (ch) => /\s/u.test(ch);

function ratio(pred, s) {
  if (!s) return 0.0;
  let n = 0;
  for (const ch of s) if (pred(ch)) n++;
  return n / s.length;
}

function safeUnquote(s) {
  try {
    return decodeURIComponent(s);
  } catch (e) {
    // Python's unquote is lenient about stray % signs; decodeURIComponent
    // throws. Fall back to a per-sequence decode so behaviour matches.
    return s.replace(/%[0-9A-Fa-f]{2}/g, (m) => {
      try { return decodeURIComponent(m); } catch (_) { return m; }
    });
  }
}

function countSub(s, sub) {
  if (!sub) return 0;
  let n = 0, i = 0;
  while ((i = s.indexOf(sub, i)) !== -1) { n++; i += sub.length; }
  return n;
}

function parseUrlParts(url) {
  // Mirrors urllib.parse.urlparse for the fields used, without throwing on
  // malformed input (Python returns empty components rather than raising).
  let scheme = "", rest = url, query = "", fragment = "";
  const sch = /^([a-zA-Z][a-zA-Z0-9+.-]*):\/\//.exec(url);
  if (sch) { scheme = sch[1].toLowerCase(); rest = url.slice(sch[0].length); }
  const hi = rest.indexOf("#");
  if (hi !== -1) { fragment = rest.slice(hi + 1); rest = rest.slice(0, hi); }
  const qi = rest.indexOf("?");
  if (qi !== -1) { query = rest.slice(qi + 1); rest = rest.slice(0, qi); }
  return { scheme, query, fragment };
}

function parseQsl(query) {
  const out = [];
  if (!query) return out;
  for (const part of query.split("&")) {
    if (!part) continue;
    const eq = part.indexOf("=");
    const k = eq === -1 ? part : part.slice(0, eq);
    const v = eq === -1 ? "" : part.slice(eq + 1);
    out.push([safeUnquote(k.replace(/\+/g, " ")), safeUnquote(v.replace(/\+/g, " "))]);
  }
  return out;
}

function urlFeatures(url) {
  const p = parseUrlParts(url);
  const query = p.query || "";
  const frag = p.fragment || "";
  const payloadArea = query + frag;
  const decoded = safeUnquote(payloadArea);
  const params = parseQsl(query);
  let longest = 0;
  for (const [, v] of params) if (v.length > longest) longest = v.length;

  return [
    Math.min(url.length / 200.0, 3.0),
    Math.min(query.length / 100.0, 3.0),
    Math.min(frag.length / 100.0, 3.0),
    Math.min(params.length / 10.0, 3.0),
    Math.min(longest / 100.0, 3.0),
    shannon(payloadArea) / 6.0,
    ratio(isDigit, url),
    ratio((c) => !isAlnum(c), url),
    Math.min(countSub(url, "%") / 20.0, 3.0),
    Math.min(countSub(url, "%25") / 5.0, 3.0),
    Math.min(countSub(url, "/") / 10.0, 3.0),
    decoded.includes("<") ? 1.0 : 0.0,
    decoded.includes(">") ? 1.0 : 0.0,
    new RegExp(DANGEROUS_TAGS.source, "i").test(decoded) ? 1.0 : 0.0,
    new RegExp(EVENT_HANDLER.source, "i").test(decoded) ? 1.0 : 0.0,
    JS_URI.test(decoded) ? 1.0 : 0.0,
    p.scheme === "https" ? 1.0 : 0.0,
    frag ? 1.0 : 0.0,
  ];
}

function scriptFeatures(script) {
  const s = script || "";
  const hasSource = SOURCE_READ.test(s);
  const hasSink = SINK_CALL.test(s);
  const low = s.toLowerCase();
  return [
    Math.min(s.length / 200.0, 3.0),
    shannon(s) / 6.0,
    ratio((c) => !isAlnum(c) && !isSpace(c), s),
    ratio(isUpper, s),
    Math.min(countMatches(DANGEROUS_TAGS, s) / 5.0, 3.0),
    Math.min(countMatches(EVENT_HANDLER, s) / 5.0, 3.0),
    JS_URI.test(s) ? 1.0 : 0.0,
    hasSink ? 1.0 : 0.0,
    hasSource ? 1.0 : 0.0,
    hasSource && hasSink ? 1.0 : 0.0,
    ALERTISH.test(s) ? 1.0 : 0.0,
    Math.min(countSub(s, "(") / 10.0, 3.0),
    Math.min(countSub(s, "`") / 5.0, 3.0),
    Math.min(countSub(s, "&#") / 5.0, 3.0),
    Math.min(countSub(s, "\\x") / 5.0, 3.0),
    (low.includes("sanitize") || low.includes("escapehtml")) ? 1.0 : 0.0,
  ];
}

function hashedNgrams(text, n = 3) {
  const vec = new Float32Array(N_HASH);
  if (!text) return vec;
  const t = text.slice(0, 600);
  for (let i = 0; i + n <= t.length; i++) {
    let h = 2166136261;
    for (let k = 0; k < n; k++) {
      h ^= t.charCodeAt(i + k);
      h = Math.imul(h, 16777619) >>> 0;   // 32-bit unsigned, as in Python
    }
    vec[h % N_HASH] += ((h >>> 31) & 1) ? 1.0 : -1.0;
  }
  let norm = 0;
  for (let i = 0; i < N_HASH; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < N_HASH; i++) vec[i] /= norm;
  return vec;
}

function featurise(record) {
  const url = record.url || "";
  const script = record.script || "";
  const v = new Float32Array(DIM);

  const uf = urlFeatures(url);
  for (let i = 0; i < 18; i++) v[i] = uf[i];
  const sf = scriptFeatures(script);
  for (let i = 0; i < 16; i++) v[18 + i] = sf[i];

  const sink = record.sink || "none";
  const si = SINKS.indexOf(sink);
  if (si >= 0) v[34 + si] = 1.0;
  v[44] = record.origin === "same" ? 1.0 : 0.0;
  v[45] = record.origin === "cross" ? 1.0 : 0.0;

  const combined = safeUnquote(url) + " " + script;
  const hv = hashedNgrams(combined);
  for (let i = 0; i < N_HASH; i++) v[OFFSET_HASH + i] = hv[i];
  return v;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { featurise, DIM, N_HASH, OFFSET_HASH, SINKS };
}
