import { useState } from "react";

type AskResponse = {
  answer: string;
  sources: string[];
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleAsk() {
    setLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data: AskResponse = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }

  async function handleAskStream() {
  setLoading(true);
  setError(null);
  setAnswer("");
  setSources([]);

  try {
    const res = await fetch("http://localhost:8000/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";        // le fragment incomplet

      for (const frame of frames) {
        if (!frame.trim()) continue;

        const eventMatch = frame.match(/^event: (.+)$/m);
        const dataMatch = frame.match(/^data: (.+)$/m);
        if (!dataMatch) continue;

        const payload = JSON.parse(dataMatch[1]);
        const eventName = eventMatch?.[1];

        if (eventName === "sources") {
          setSources(payload);
        } else if (eventName === "done") {
          // rien à faire, la boucle va se terminer
        } else {
          setAnswer((prev) => prev + payload.token);
        }
      }
    }
  } catch (e) {
    setError(e instanceof Error ? e.message : "Erreur inconnue");
  } finally {
    setLoading(false);
  }
}

  return (
    <main>
      <h1>Garden Copilot</h1>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={4}
        placeholder="Quand tailler les tomates ?"
      />

      <button onClick={handleAsk} disabled={loading || !question.trim()}>
        {loading ? "Recherche…" : "Demander"}
      </button>

      {error && <p role="alert">Erreur : {error}</p>}

      {answer && (
        <section>
          <p>{answer}</p>
          <ul>
            {sources.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}