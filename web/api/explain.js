// Vercel serverless function: takes a transaction's real SHAP drivers (computed offline by the
// Python pipeline, shipped in data/demo_transactions.json) and asks Gemini, live, to translate
// them into plain language. Mirrors src/explain_gemini.py's approach exactly, so the deployed
// demo and the local pipeline share one design: SHAP computes the reasons, Gemini only rephrases
// the reasons it's given — grounded, not free-associated.

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const SYSTEM_INSTRUCTION =
  "You are FraudLens' explanation layer. You are given a fraud model's output probability and " +
  "its top SHAP feature attributions for one transaction. Write a 2-4 sentence plain-language " +
  "explanation for a fraud analyst who is not a data scientist. Rules: only reference the " +
  "features and directions given to you — never invent a reason not present in the data. State " +
  "the risk level and probability plainly, then explain the top 2-3 drivers in business terms. " +
  "Do not use technical jargon like 'SHAP value' or 'feature vector' in the output. Do not use " +
  "markdown formatting (no asterisks, no bold) — plain sentences only. IMPORTANT: engineered " +
  "features named C1-C14, D1-D15, and V1-V339 are anonymized by the data provider and their " +
  "real-world meaning was never disclosed — never claim to know what one of these specifically " +
  "represents (e.g. never say a C-feature counts 'purchases from the same IP' or a V-feature " +
  "means something specific). For these, describe only the direction and relative size of the " +
  "effect (e.g. 'an unusual value in one of the model's internal risk signals'), and reserve " +
  "concrete real-world claims for named fields you can be certain about, like card network, " +
  "email domain, device, or transaction amount.";

function formatDrivers(topDrivers) {
  return topDrivers
    .map((d) => {
      const direction = d.shap > 0 ? "pushes toward FRAUD" : "pushes toward LEGITIMATE";
      return `- ${d.feature} = ${d.value}  (${direction}, strength ${Math.abs(d.shap).toFixed(3)})`;
    })
    .join("\n");
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Use POST" });
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "GEMINI_API_KEY not configured on the server" });
    return;
  }

  const { fraudProbability, topDrivers } = req.body || {};
  if (typeof fraudProbability !== "number" || !Array.isArray(topDrivers)) {
    res.status(400).json({ error: "Expected { fraudProbability: number, topDrivers: array }" });
    return;
  }

  const prompt =
    `Fraud probability: ${(fraudProbability * 100).toFixed(1)}%\n` +
    `Top feature drivers (sorted by influence):\n${formatDrivers(topDrivers)}\n\n` +
    "Write the plain-language explanation now.";

  const payload = {
    system_instruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 400,
      thinkingConfig: { thinkingBudget: 0 },
    },
  };

  try {
    const resp = await fetch(`${GEMINI_URL}?key=${encodeURIComponent(apiKey)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      res.status(502).json({ error: `Gemini API error: ${resp.status}`, detail: errText });
      return;
    }

    const data = await resp.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    if (!text) {
      res.status(502).json({ error: "Gemini returned no text", raw: data });
      return;
    }

    res.status(200).json({ explanation: text });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
}
