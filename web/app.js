let transactions = [];

async function init() {
  const res = await fetch("data/demo_transactions.json");
  transactions = await res.json();
  renderList();
  if (transactions.length) selectTransaction(transactions[0].transactionId);
}

function renderList() {
  const list = document.getElementById("txn-list");
  list.innerHTML = "";
  for (const t of transactions) {
    const btn = document.createElement("button");
    btn.className = "txn-card";
    btn.dataset.id = t.transactionId;
    btn.innerHTML = `
      <div class="id">Txn ${t.transactionId}</div>
      <div class="prob">${(t.fraudProbability * 100).toFixed(1)}%</div>
      <div class="label ${t.actualLabel}">${t.actualLabel === "fraud" ? "Actual: Fraud" : "Actual: Legit"}</div>
    `;
    btn.addEventListener("click", () => selectTransaction(t.transactionId));
    list.appendChild(btn);
  }
}

function selectTransaction(id) {
  const t = transactions.find((x) => x.transactionId === id);
  if (!t) return;

  document.querySelectorAll(".txn-card").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.id) === id);
  });

  const detail = document.getElementById("detail");
  detail.hidden = false;

  document.getElementById("d-id").textContent = t.transactionId;
  const badge = document.getElementById("d-actual");
  badge.textContent = t.actualLabel === "fraud" ? "Actual: Fraud" : "Actual: Legit";
  badge.className = `badge ${t.actualLabel}`;

  document.getElementById("d-prob").textContent = `${(t.fraudProbability * 100).toFixed(1)}%`;
  document.getElementById("d-prob").style.color = t.fraudProbability > 0.5 ? "#c0392b" : "#1a7a6e";

  renderDrivers(t.topDrivers);
  fetchExplanation(t);
}

function renderDrivers(drivers) {
  const el = document.getElementById("d-drivers");
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap)), 0.001);
  el.innerHTML = drivers
    .map((d) => {
      const pct = (Math.abs(d.shap) / maxAbs) * 100;
      const cls = d.shap > 0 ? "pos" : "neg";
      return `
        <div class="driver-row">
          <div class="driver-name">${d.feature} = ${d.value}</div>
          <div class="driver-bar-track"><div class="driver-bar ${cls}" style="width:${pct}%"></div></div>
        </div>
      `;
    })
    .join("");
}

async function fetchExplanation(t) {
  const el = document.getElementById("d-explanation");
  el.innerHTML = '<span class="loading">Calling Gemini…</span>';
  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fraudProbability: t.fraudProbability,
        topDrivers: t.topDrivers,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "request failed");
    el.textContent = data.explanation;
  } catch (err) {
    el.innerHTML = `<span class="error">Explanation call failed: ${err.message}</span>`;
  }
}

init();
