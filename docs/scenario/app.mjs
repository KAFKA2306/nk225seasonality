const form = document.querySelector("#scenario-form");
const output = document.querySelector("#scenario-output");
const runtime = document.querySelector("#runtime");
const manifestNode = document.querySelector("#manifest");

const manifest = await fetch("./scenario.json").then((response) => {
  if (!response.ok) throw new Error(`scenario manifest unavailable: ${response.status}`);
  return response.json();
});
manifestNode.textContent = [
  `mode: ${manifest.mode}`,
  `data as-of: ${manifest.data_as_of ?? "not loaded"}`,
  `JGB observed_at: ${manifest.jgb_observed_at ?? "not loaded"}`,
  `method: ${manifest.method}`,
].join(" · ");

const worker = new Worker("./worker.mjs", { type: "module" });
let requestId = 0;
const pending = new Map();

worker.onmessage = ({ data }) => {
  const resolve = pending.get(data.requestId);
  if (!resolve) return;
  pending.delete(data.requestId);
  resolve(data);
};

function runScenario(payload) {
  return new Promise((resolve) => {
    requestId += 1;
    pending.set(requestId, resolve);
    worker.postMessage({ requestId, payload });
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  output.textContent = "Calculating in Python…";
  const payload = Object.fromEntries(
    ["current_per", "jgb_yield", "risk_premium"].map((name) => [name, Number(form.elements[name].value)]),
  );
  const response = await runScenario(payload);
  if (!response.ok) {
    output.textContent = `Blocked: ${response.error}`;
    return;
  }
  const result = response.result;
  output.textContent = [
    `Fair PER: ${result.fair_per.toFixed(2)}x`,
    `Earnings yield: ${result.earnings_yield.toFixed(2)}%`,
    `Yield gap: ${result.yield_gap.toFixed(2)}pt`,
    `Divergence: ${result.divergence_pct.toFixed(2)}%`,
    `Method: ${result.method}`,
  ].join("\n");
  runtime.textContent = `Pyodide initial load: ${response.loadMs} ms · calculation: ${response.calculationMs} ms`;
});
