/**
 * QuietHorizon – Static GitHub Pages Front End
 *
 * Sends an uploaded audio file to the AWS API Gateway endpoint and
 * renders the classification result without any server-side Python.
 *
 * ── Configuration ─────────────────────────────────────────────
 * Set API_ENDPOINT to your deployed AWS API Gateway / Lambda URL.
 * The endpoint must accept:
 *   POST  multipart/form-data  { "file": <audio blob>, "filename": <string> }
 * and return JSON:
 *   {
 *     "predicted_label": "nature" | "anthro",
 *     "prob_nature":     <float 0–1>,
 *     "prob_anthro":     <float 0–1>,
 *     "confidence":      "HIGH" | "MEDIUM" | "LOW"   (optional)
 *   }
 * ──────────────────────────────────────────────────────────────
 */

const API_ENDPOINT = "https://YOUR_API_GATEWAY_URL/analyze"; // ← replace this

const MAX_FILE_BYTES = 50 * 1024 * 1024; // 50 MB
const ACCEPTED_TYPES = new Set(["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/x-m4a", "audio/mp4", "audio/x-wav"]);
const ACCEPTED_EXTS  = new Set([".wav", ".mp3", ".ogg", ".flac", ".m4a"]);

// ─── Element refs ──────────────────────────────────────────────
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const fileInfo      = document.getElementById("file-info");
const fileName      = document.getElementById("file-name");
const fileSize      = document.getElementById("file-size");
const removeBtn     = document.getElementById("remove-file");
const audioPreview  = document.getElementById("audio-preview");
const audioPlayer   = document.getElementById("audio-player");
const analyzeBtn    = document.getElementById("analyze-btn");
const btnText       = document.getElementById("btn-text");
const btnSpinner    = document.getElementById("btn-spinner");
const errorBanner   = document.getElementById("error-banner");
const resultsSection = document.getElementById("results-section");
const verdictEl     = document.getElementById("verdict");
const verdictIcon   = document.getElementById("verdict-icon");
const verdictLabel  = document.getElementById("verdict-label");
const verdictFile   = document.getElementById("verdict-file");
const confidenceBadge = document.getElementById("confidence-badge");
const barNature     = document.getElementById("bar-nature");
const barAnthro     = document.getElementById("bar-anthro");
const pctNature     = document.getElementById("pct-nature");
const pctAnthro     = document.getElementById("pct-anthro");
const rawJsonPre    = document.getElementById("raw-json-pre");

let selectedFile = null;
let objectUrl    = null;

// ─── Drag-and-drop ────────────────────────────────────────────
["dragenter", "dragover"].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add("dragover"); })
);

["dragleave", "drop"].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove("dragover"); })
);

dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});

// Keyboard activation for drop zone
dropZone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

// ─── File input change ────────────────────────────────────────
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
});

// ─── Remove file ──────────────────────────────────────────────
removeBtn.addEventListener("click", clearFile);

// ─── Analyze button ───────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

// ─── Handle a chosen file ─────────────────────────────────────
function handleFile(file) {
  hideError();

  // Extension check
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const mimeOk = ACCEPTED_TYPES.has(file.type);
  const extOk  = ACCEPTED_EXTS.has(ext);

  if (!mimeOk && !extOk) {
    showError(`Unsupported file type "${ext}". Please upload WAV, MP3, OGG, FLAC, or M4A.`);
    return;
  }

  if (file.size > MAX_FILE_BYTES) {
    showError(`File is too large (${formatBytes(file.size)}). Maximum allowed size is 50 MB.`);
    return;
  }

  selectedFile = file;

  // Revoke previous object URL
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);

  // Update file info bar
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.classList.remove("hidden");

  // Audio preview
  audioPlayer.src = objectUrl;
  audioPreview.classList.remove("hidden");

  analyzeBtn.disabled = false;

  // Hide old results
  resultsSection.classList.add("hidden");
}

// ─── Clear selection ──────────────────────────────────────────
function clearFile() {
  selectedFile = null;
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  fileInput.value = "";
  fileInfo.classList.add("hidden");
  audioPreview.classList.add("hidden");
  audioPlayer.src = "";
  analyzeBtn.disabled = true;
  hideError();
}

// ─── Run analysis ─────────────────────────────────────────────
async function runAnalysis() {
  if (!selectedFile) return;

  // Safety guard: warn if endpoint hasn't been configured
  if (API_ENDPOINT.includes("YOUR_API_GATEWAY_URL")) {
    showError(
      "API endpoint not configured. " +
      "Open docs/app.js and set API_ENDPOINT to your AWS API Gateway URL."
    );
    return;
  }

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append("file", selectedFile, selectedFile.name);
  formData.append("filename", selectedFile.name);

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        detail = errJson.detail || errJson.message || detail;
      } catch (_) { /* body was not JSON */ }
      throw new Error(detail);
    }

    const data = await response.json();
    renderResults(data, selectedFile.name);

  } catch (err) {
    showError(`Analysis failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ─── Render results ───────────────────────────────────────────
function renderResults(data, filename) {
  const label      = (data.predicted_label || "").toLowerCase();
  const probNature = clamp(parseFloat(data.prob_nature ?? 0));
  const probAnthro = clamp(parseFloat(data.prob_anthro ?? 0));
  const confidence = (data.confidence || (Math.max(probNature, probAnthro) >= 0.85 ? "HIGH" : "MEDIUM")).toUpperCase();

  // Verdict block
  const isNature = label === "nature";
  verdictEl.className = "verdict " + (isNature ? "nature" : "anthro");
  verdictIcon.textContent = isNature ? "🌿" : "🏭";
  verdictLabel.textContent = isNature ? "Natural Soundscape" : "Anthropogenic Noise";
  verdictFile.textContent = filename;
  confidenceBadge.textContent = confidence + " confidence";

  // Probability bars (animate after a tiny delay so transition fires)
  requestAnimationFrame(() => {
    const nPct = (probNature * 100).toFixed(1);
    const aPct = (probAnthro * 100).toFixed(1);

    barNature.style.width = nPct + "%";
    barAnthro.style.width = aPct + "%";
    pctNature.textContent = nPct + "%";
    pctAnthro.textContent = aPct + "%";

    // Accessibility
    barNature.closest("[role=progressbar]").setAttribute("aria-valuenow", nPct);
    barAnthro.closest("[role=progressbar]").setAttribute("aria-valuenow", aPct);
  });

  // Raw JSON
  rawJsonPre.textContent = JSON.stringify(data, null, 2);

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── Loading state ────────────────────────────────────────────
function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.textContent = loading ? "Analyzing…" : "Analyze";
  btnSpinner.classList.toggle("hidden", !loading);
}

// ─── Error helpers ────────────────────────────────────────────
function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

// ─── Utilities ────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function clamp(v) { return Math.min(1, Math.max(0, isNaN(v) ? 0 : v)); }
