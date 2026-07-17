/* InsideAI end-to-end UI verification (playwright-core + system Edge). */
const { chromium } = require("playwright-core");
const path = require("path");

const SHOTS = __dirname;
const URL = "http://localhost:3000";
const step = (s) => console.log("STEP  " + s);
const ok = (s) => console.log("OK    " + s);

let browser;
let page;
const problems = [];

(async () => {
  browser = await chromium.launch({
    channel: "msedge",
    headless: true,
    args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"],
  });
  // Modest viewport: headless runs on SwiftShader (software GL), and this
  // machine is often RAM-starved — keep the test load realistic.
  page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") problems.push("console.error: " + m.text());
  });

  step("load " + URL);
  await page.goto(URL, { waitUntil: "domcontentloaded" });

  step("wait for model_info over websocket (model chip)");
  await page.waitForSelector("text=/\\d+ layers/", { timeout: 30000 });
  const chip = await page.textContent("text=/\\d+ layers/");
  ok(`model chip shows "${chip?.trim()}" (WS model_info applied)`);

  await page.waitForSelector("text=live", { timeout: 15000 });
  ok("connection status: live");

  const canvases = await page.evaluate(() =>
    Array.from(document.querySelectorAll("canvas")).map((c) => `${c.width}x${c.height}`)
  );
  if (canvases.length < 1) throw new Error("no canvas found — 3D scene missing");
  ok(`canvases present: ${canvases.join(", ")} (scene + heatmap)`);

  step("PROBE: empty prompt → Generate must be disabled");
  const disabled = await page.$eval("button:has-text('Generate')", (b) => b.disabled);
  if (!disabled) throw new Error("Generate enabled with empty prompt");
  ok("probe: Generate disabled while prompt is empty");

  step("open settings, set pacing to Instant, 12 tokens");
  await page.click("button[title='Generation settings']");
  await page.click("button:has-text('Instant')");
  await page.locator("input[type='range']").first().fill("12"); // new-tokens slider
  ok("pacing = Instant, max_new_tokens = 12");

  step("type prompt and generate");
  await page.fill("textarea", "Explain quantum computing");
  await page.click("button:has-text('Generate')");

  step("wait for generated token chips (websocket stream driving UI)");
  await page.waitForFunction(
    () => document.querySelectorAll("[title*='rank']").length >= 6,
    undefined,
    { timeout: 180000 }
  );
  const chipCount = await page.evaluate(
    () => document.querySelectorAll("[title*='rank']").length
  );
  ok(`generated token chips visible: ${chipCount}`);

  await page.waitForSelector("text=entropy", { timeout: 30000 });
  ok("probability panel live (entropy readout present)");

  step("wait for run to finish (button returns to Generate)");
  await page.waitForSelector("button:has-text('Generate')", { timeout: 360000 });
  const bodyText = await page.evaluate(() => document.body.innerText);
  if (!/entropy\s+[\d.]+\s+bits/i.test(bodyText)) throw new Error("no entropy value rendered");
  ok("entropy value rendered: " + (bodyText.match(/entropy\s+[\d.]+\s+bits/i) || [""])[0]);
  await page.screenshot({ path: path.join(SHOTS, "shot1-instant-done.png") });
  ok("screenshot shot1-instant-done.png");

  step("PROBE: cinematic run, then Stop mid-generation");
  // exact:true avoids the TopBar "✦ Cinematic" camera toggle
  await page.getByRole("button", { name: "Cinematic", exact: true }).click();
  await page.fill("textarea", "Once upon a time, a robot");
  await page.click("button:has-text('Generate')");
  await page.waitForSelector("button:has-text('Stop')", { timeout: 15000 });
  await page.waitForTimeout(5000); // first-step ceremony ≈10s, still mid-run
  await page.screenshot({ path: path.join(SHOTS, "shot2-midrun.png") });
  ok("screenshot shot2-midrun.png (mid-generation)");
  await page.click("button:has-text('Stop')");
  await page.waitForSelector("text=cancelled", { timeout: 20000 });
  ok("probe: Stop → 'cancelled' badge appears in token stream");

  step("PROBE: immediately start a fresh run after cancel");
  await page.fill("textarea", "The moon is");
  await page.click("button:has-text('Instant')");
  await page.click("button:has-text('Generate')");
  // The regression target is a clean restart: the run begins (Stop appears)
  // and no error toast shows — a full extra generation isn't needed.
  await page.waitForSelector("button:has-text('Stop')", { timeout: 30000 });
  await page.waitForTimeout(8000);
  const errToast = await page.$("text=error ·");
  if (errToast) throw new Error("error toast appeared on immediate re-run");
  ok("probe: immediate re-run after cancel starts cleanly, no error toast");
  await page.screenshot({ path: path.join(SHOTS, "shot3-rerun.png") });
  ok("screenshot shot3-rerun.png");
  await page.click("button:has-text('Stop')");
  await page.waitForSelector("button:has-text('Generate')", { timeout: 60000 });
  ok("probe: second Stop also lands cleanly");

  const sceneInfo = await page.evaluate(() => {
    const c = document.querySelector("canvas");
    const gl = c && (c.getContext("webgl2") || c.getContext("webgl"));
    return { webgl: !!gl, renderer: gl ? gl.getParameter(gl.RENDERER) : null };
  });
  console.log("INFO  webgl:", JSON.stringify(sceneInfo));

  console.log("\nPROBLEMS (console/page errors):", problems.length ? "" : "none");
  for (const p of problems) console.log("  " + p);

  await browser.close();
  if (problems.some((p) => p.includes("pageerror"))) {
    console.log("E2E RESULT: FAIL (page errors)");
    process.exit(1);
  }
  console.log("E2E RESULT: PASS");
})().catch(async (e) => {
  console.error("E2E RESULT: FAIL —", e.message);
  if (problems.length) {
    console.error("collected page problems:");
    for (const p of problems) console.error("  " + p);
  }
  try {
    if (page) await page.screenshot({ path: path.join(SHOTS, "fail.png") });
    console.error("failure screenshot: scripts/fail.png");
  } catch {}
  try {
    if (browser) await browser.close();
  } catch {}
  process.exit(1);
});
