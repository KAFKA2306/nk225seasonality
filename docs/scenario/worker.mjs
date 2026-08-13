import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs";

let runtimePromise;

async function getRuntime() {
  if (!runtimePromise) {
    const started = performance.now();
    runtimePromise = (async () => {
      const pyodide = await loadPyodide();
      const pythonSource = await fetch("./browser_scenario.py").then((response) => {
        if (!response.ok) throw new Error(`python source unavailable: ${response.status}`);
        return response.text();
      });
      pyodide.FS.writeFile("/browser_scenario.py", pythonSource);
      pyodide.runPython("import sys; sys.path.insert(0, '/')");
      return { pyodide, loadMs: performance.now() - started };
    })();
  }
  return runtimePromise;
}

self.onmessage = async ({ data }) => {
  const requestId = data?.requestId;
  try {
    const { pyodide, loadMs } = await getRuntime();
    const started = performance.now();
    pyodide.globals.set("scenario_payload", JSON.stringify(data.payload));
    const result = pyodide.runPython(`
from browser_scenario import calculate_scenario_json
calculate_scenario_json(scenario_payload)
`);
    self.postMessage({
      requestId,
      ok: true,
      result: JSON.parse(result),
      loadMs: Math.round(loadMs),
      calculationMs: Math.round((performance.now() - started) * 100) / 100,
    });
  } catch (error) {
    self.postMessage({ requestId, ok: false, error: String(error?.message ?? error) });
  }
};
