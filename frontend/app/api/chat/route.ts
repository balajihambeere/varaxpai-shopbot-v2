import { NextRequest, NextResponse } from "next/server";

// Points at the catalog-retrieval-assistant backend (book-2/app/backend),
// not the original backend-shopbot-api. Its endpoint is POST /v1/chat with
// body {session_id, query} — this route adapts ChatBot.tsx's existing
// {question, session_id} shape so the component itself needed no changes.
const BACKEND_URL = process.env.SHOPBOT_API_URL ?? "http://localhost:8010";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { question, session_id } = body;

    const res = await fetch(`${BACKEND_URL}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, session_id }),
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { answer: `Backend error (${res.status}): ${text}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      {
        answer:
          "The assistant is not reachable. Make sure the backend is running on port 8010 (docker compose up, from book-2/app/backend).",
        error: message,
      },
      { status: 503 }
    );
  }
}
