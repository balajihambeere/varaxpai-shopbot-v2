// Records real browser interaction with the live ShopBot chat widget.
// Requires: backend running on :8000 with a real OPENAI_API_KEY, frontend
// running on :3000 (npm run dev in zudyog-fashion). Produces a real .webm
// video via Playwright's native recordVideo — no screenshots, no blind
// clicking; every step waits for and asserts on the actual rendered answer
// text before moving on.
import { chromium } from "playwright";
import { mkdirSync } from "fs";

const OUT_DIR = "/Applications/workplace/rag-series/rag-essentials/docs/video/raw";
mkdirSync(OUT_DIR, { recursive: true });

const EXTENDED = process.argv.includes("--extended");

async function ask(page, question) {
  const input = page.getByPlaceholder("Ask anything about our collection…");
  await input.fill(question);
  await input.press("Enter");
}

async function run() {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: OUT_DIR, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();

  await page.goto("http://localhost:3000", { waitUntil: "networkidle" });
  await page.waitForTimeout(800); // let the hook's "text on screen" beat breathe

  await page.getByRole("button", { name: "Open AI stylist" }).click();
  await page.waitForTimeout(500);

  // Shot 1 — grounded answer (real, verified example from shopbot/README.md)
  await ask(page, "Does the anarkali suit come in size XL?");
  await page.getByText(/teal/i).waitFor({ timeout: 20_000 });
  await page.waitForTimeout(1500); // let the real answer sit on screen

  // Shot 2 — honest refusal (real fallback, from evaluation/test_cases.py)
  await ask(page, "Do you sell men's sherwanis?");
  await page.getByText(/support@zudyog\.com/i).waitFor({ timeout: 20_000 });
  await page.waitForTimeout(1500);

  if (EXTENDED) {
    // Shot 3 — the real multi-turn memory gap (Book 1's honest limitation)
    await ask(page, "Does the Breathable Cotton Kurta come in blue?");
    await page.getByText(/sky blue/i).waitFor({ timeout: 20_000 });
    await page.waitForTimeout(1200);

    await ask(page, "Does it come in blue?");
    await page.waitForTimeout(4000); // no assertion here on purpose —
    // whatever it actually answers (right or wrong) is the real, honest result
  }

  await context.close(); // finalizes the .webm file
  await browser.close();
  console.log(`Video saved to ${OUT_DIR}`);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
