const SAMPLES = {
  cardinal: {
    title: "Northern Cardinal",
    kind: "Repository sample",
    audio: "samples/northern-cardinal.wav",
    result: "samples/northern-cardinal.json",
  },
  traffic: {
    title: "Heavy Traffic",
    kind: "Repository sample",
    audio: "samples/heavy-traffic.wav",
    result: "samples/heavy-traffic.json",
  },
};

const sampleGrid = document.getElementById("sample-grid");
const resultsSection = document.getElementById("results-section");
const resultTitle = document.getElementById("result-title");
const resultKind = document.getElementById("result-kind");
const audioPlayer = document.getElementById("audio-player");
const verdictEl = document.getElementById("verdict");
const verdictIcon = document.getElementById("verdict-icon");
const verdictLabel = document.getElementById("verdict-label");
const verdictFile = document.getElementById("verdict-file");
const confidenceBadge = document.getElementById("confidence-badge");
const barNature = document.getElementById("bar-nature");
const barAnthro = document.getElementById("bar-anthro");
const pctNature = document.getElementById("pct-nature");
const pctAnthro = document.getElementById("pct-anthro");
const rawJsonPre = document.getElementById("raw-json-pre");

sampleGrid.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-sample-id]");
  if (!button) return;

  const sampleId = button.getAttribute("data-sample-id");
  const sample = SAMPLES[sampleId];
  if (!sample) return;

  setActiveSample(sampleId);
  await loadSample(sample);
});

async function loadSample(sample) {
  try {
    const response = await fetch(sample.result);
    if (!response.ok) {
      throw new Error(`Unable to load stored result (${response.status})`);
    }

    const data = await response.json();
    renderSample(sample, data);
  } catch (error) {
    rawJsonPre.textContent = `Failed to load demo result: ${error.message}`;
    resultsSection.classList.remove("hidden");
  }
}

function renderSample(sample, data) {
  const label = (data.predicted_label || "").toLowerCase();
  const probNature = clamp(data.prob_nature);
  const probAnthro = clamp(data.prob_anthro);
  const isNature = label === "nature";

  resultTitle.textContent = sample.title;
  resultKind.textContent = sample.kind;
  audioPlayer.src = sample.audio;
  audioPlayer.load();

  verdictEl.className = `verdict ${isNature ? "nature" : "anthro"}`;
  verdictIcon.textContent = isNature ? "N" : "A";
  verdictLabel.textContent = isNature ? "Natural soundscape" : "Anthropogenic noise";
  verdictFile.textContent = data.source_file || sample.audio;
  confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const naturePct = (probNature * 100).toFixed(1);
  const anthroPct = (probAnthro * 100).toFixed(1);

  barNature.style.width = `${naturePct}%`;
  barAnthro.style.width = `${anthroPct}%`;
  pctNature.textContent = `${naturePct}%`;
  pctAnthro.textContent = `${anthroPct}%`;
  barNature.closest("[role=progressbar]").setAttribute("aria-valuenow", naturePct);
  barAnthro.closest("[role=progressbar]").setAttribute("aria-valuenow", anthroPct);

  rawJsonPre.textContent = JSON.stringify(data, null, 2);
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setActiveSample(sampleId) {
  const cards = sampleGrid.querySelectorAll("[data-sample-id]");
  cards.forEach((card) => {
    card.classList.toggle("is-active", card.getAttribute("data-sample-id") === sampleId);
  });
}

function clamp(value) {
  const numeric = Number.parseFloat(value);
  if (Number.isNaN(numeric)) return 0;
  return Math.min(1, Math.max(0, numeric));
}
