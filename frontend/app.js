// frontend/app.js

const API_BASE = window.location.origin;
const resultsList = document.getElementById("results-list");
const resultsInfo = document.getElementById("results-info");
const experimentOutput = document.getElementById("experiment-output");

async function doSearch() {
  const query = document.getElementById("query").value.trim();
  const tokenizer = document.getElementById("tokenizer").value;
  const removeStopwords = document.getElementById("remove_stopwords").checked;
  const useStemming = document.getElementById("use_stemming").checked;
  const indexType = document.getElementById("index_type").value;
  const topK = parseInt(document.getElementById("top_k").value, 10);

  if (!query) {
    resultsInfo.textContent = "Please paste a job description to kick off the search.";
    resultsList.innerHTML = "";
    return;
  }

  const payload = {
    query,
    tokenizer_name: tokenizer,
    remove_stopwords: removeStopwords,
    use_stemming: useStemming,
    index_type: indexType,
    top_k: topK
  };

  resultsInfo.textContent = "Searching resumes...";
  resultsList.innerHTML = "";

  try {
    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Search failed (${res.status})`);
    }

    const data = await res.json();
    renderResults(data.results, topK);
  } catch (err) {
    resultsInfo.textContent = err.message;
    console.error(err);
  }
}

function renderResults(results, requestedK) {
  resultsList.innerHTML = "";

  if (!results || results.length === 0) {
    resultsInfo.textContent = "No resumes matched this description. Try relaxing filters or reducing required skills.";
    return;
  }

  resultsInfo.textContent = `Showing ${results.length} of the requested top ${requestedK} resumes.`;

  results.forEach((r, idx) => {
    const article = document.createElement("article");
    article.className = "result-card";

    const header = document.createElement("div");
    header.className = "result-header";

    const rank = document.createElement("div");
    rank.className = "result-rank";
    rank.textContent = `#${idx + 1}`;

    const meta = document.createElement("div");
    meta.className = "result-meta";
    const percentScore = Math.round(r.score * 10000) / 100;
    meta.innerHTML = `<strong>Resume ID ${r.doc_id}</strong><span>${percentScore}% match</span>`;

    header.appendChild(rank);
    header.appendChild(meta);

    const snippet = document.createElement("p");
    snippet.className = "snippet";
    snippet.textContent = r.snippet || "Snippet unavailable for this resume.";

    article.appendChild(header);
    article.appendChild(snippet);
    resultsList.appendChild(article);
  });
}

async function doExperiment() {
  const raw = document.getElementById("experiment-json").value;
  const tokenizer = document.getElementById("tokenizer").value;
  const removeStopwords = document.getElementById("remove_stopwords").checked;
  const useStemming = document.getElementById("use_stemming").checked;
  const indexType = document.getElementById("index_type").value;
  const topK = parseInt(document.getElementById("top_k").value, 10);

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    alert("Invalid JSON in experiment box");
    return;
  }

  const payload = {
    ...parsed,
    tokenizer_name: tokenizer,
    remove_stopwords: removeStopwords,
    use_stemming: useStemming,
    index_type: indexType,
    top_k: topK
  };

  experimentOutput.textContent = "Running experiment...";

  try {
    const res = await fetch(`${API_BASE}/experiment`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Experiment failed (${res.status})`);
    }

    const data = await res.json();
    experimentOutput.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    experimentOutput.textContent = err.message;
    console.error(err);
  }
}

document.getElementById("search-btn").addEventListener("click", doSearch);
document.getElementById("experiment-btn").addEventListener("click", doExperiment);
