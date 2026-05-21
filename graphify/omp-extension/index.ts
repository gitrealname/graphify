import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";


// Respects GRAPHIFY_OUT env var — mirrors Python: os.environ.get("GRAPHIFY_OUT", "graphify-out")
function graphifyOut(): string { return process.env.GRAPHIFY_OUT ?? "graphify-out"; }


// ── Python interpreter detection ──────────────────────────────────────────────
// Mirrors skill Step 1: read shebang from graphify binary, fall back to python3.
// Result cached in <graphify-out>/.graphify_python (matches skill convention).
let _pythonCache: string | null = null;

function detectPython(): string {
    if (_pythonCache) return _pythonCache;

    // Check if a previous graphify run already wrote the interpreter path.
    const cached = join(process.cwd(), graphifyOut(), ".graphify_python");
    if (existsSync(cached)) {
        const p = readFileSync(cached, "utf-8").trim();
        if (p) { _pythonCache = p; return p; }
    }

    // Find graphify binary and read its shebang.
    let python = "python3";
    const graphifyBin = Bun.which("graphify");
    if (graphifyBin) {
        try {
            const firstLine = readFileSync(graphifyBin, "utf-8").split("\n")[0];
            if (firstLine.startsWith("#!")) {
                const candidate = firstLine.slice(2).trim();
                // Only accept clean paths (no special chars except / \ . - _)
                if (/^[a-zA-Z0-9/_\\.:-]+$/.test(candidate)) {
                    python = candidate;
                }
            }
        } catch { /* fall through to default */ }
    }

    // Persist for subsequent steps (matches skill convention).
    try {
        require("node:fs").mkdirSync(join(process.cwd(), graphifyOut()), { recursive: true });
        writeFileSync(cached, python, "utf-8");
    } catch { /* non-fatal */ }

    _pythonCache = python;
    return python;
}

type AutocompleteItem = { label: string; value: string };

// ── shell-aware arg split ─────────────────────────────────────────────────────

function shellSplit(input: string): string[] {
    const args: string[] = [];
    let current = "";
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < input.length; i++) {
        const ch = input[i];
        if (ch === "'" && !inDouble) { inSingle = !inSingle; }
        else if (ch === '"' && !inSingle) { inDouble = !inDouble; }
        else if (ch === " " && !inSingle && !inDouble) {
            if (current) { args.push(current); current = ""; }
        } else { current += ch; }
    }
    if (current) args.push(current);
    return args;
}

// ── Anthropic proxy server ────────────────────────────────────────────────────

const MAX_CONCURRENT = 5;

// Enhanced extraction system prompt — matches the graphify skill's Step B2 rules.
// deepMode=true adds aggressive INFERRED edges (--mode deep flag).
function buildExtractionSystem(deepMode: boolean): string {
    return `You are a graphify semantic extraction agent. Extract a knowledge graph fragment from the files provided.
Output ONLY valid JSON — no explanation, no markdown fences, no preamble.

Files are separated by === path === markers. Process each file section independently, then merge all results into the single JSON output.

Rules:
- EXTRACTED: relationship explicit in source (import, call, citation, reference)
- INFERRED: reasonable inference (shared structure, implied dependency)
- AMBIGUOUS: uncertain — flag it, do not omit
- confidence_score REQUIRED on every edge: EXTRACTED=1.0, INFERRED=0.6-0.9 (reason individually), AMBIGUOUS=0.1-0.3

Code files: extract semantic edges AST cannot find. Do NOT re-extract imports or calls already captured by AST.
Doc/paper files: extract named concepts, entities, citations. Use file_type "rationale" for concept-like nodes (ideas, principles, decisions). Store WHY decisions were made as a rationale_for edge, not a separate node.
Semantic similarity: if two concepts across files solve the same problem without a structural link, add a semantically_similar_to INFERRED edge (confidence 0.6-0.95). Non-obvious cross-file connections only.
Hyperedges: if 3+ nodes share a concept or flow not captured by pairwise edges, add a hyperedge. Max 3 per file.${deepMode ? `

DEEP MODE: be aggressive with INFERRED edges. Pursue every reasonable inference. Add semantically_similar_to edges for concepts that solve the same problem even if only loosely related. Prefer more edges over fewer.` : ""}

Node ID format: lowercase, only [a-z0-9_]. Format: {stem}_{entity} where stem = filename stem, entity = symbol name (both normalised).
IMPORTANT: file_type must be exactly one of: code, document, paper, image, rationale. Never use a file path as file_type.

Output exactly this schema:
{"nodes":[{"id":"stem_entity","label":"Human Readable Name","file_type":"code|document|paper|image|rationale","source_file":"relative/path","source_location":null,"source_url":null,"captured_at":null,"author":null,"contributor":null}],"edges":[{"source":"node_id","target":"node_id","relation":"calls|implements|references|cites|conceptually_related_to|shares_data_with|semantically_similar_to|rationale_for","confidence":"EXTRACTED|INFERRED|AMBIGUOUS","confidence_score":1.0,"source_file":"relative/path","source_location":null,"weight":1.0}],"hyperedges":[{"id":"snake_case_id","label":"Human Readable Label","nodes":["node_id1","node_id2","node_id3"],"relation":"participate_in|implement|form","confidence":"EXTRACTED|INFERRED","confidence_score":0.75,"source_file":"relative/path"}],"input_tokens":0,"output_tokens":0}`;
}

// Returns null if completeSimple is not available (non-corp OMP).
async function anthropicProxyServer(pi: any, ctx: any, deepMode: boolean): Promise<{ port: number; stop: () => number } | null> {
    // Access SDK via the injected pi-coding-agent module — no direct imports.
    const piModule = pi.pi;
    const { settings } = piModule;
    const logger = piModule.logger;
    // completeSimple is AWS-CORP only — exported from sdk.ts with // AWS-CORP marker.
    // Not available in upstream OMP. Check before starting the proxy.
    const completeSimple = piModule.completeSimple;
    if (!completeSimple) {
        logger.debug("[DBG proxy] completeSimple not available — proxy disabled (upstream OMP or missing export)");
        return null;
    }

    // Resolve model: honour GRAPHIFY_MODEL_ROLE, fall back to session model.
    const role = process.env.GRAPHIFY_MODEL_ROLE ?? "smol";
    const modelRoles = settings.get("modelRoles") as Record<string, string>;
    const specifier = modelRoles[role] ?? modelRoles["smol"];
    const available: any[] = ctx.modelRegistry.getAvailable();
    const model = (specifier ? available.find((m: any) => m.id === specifier) : null) ?? ctx.model;

    logger.debug(`[DBG proxy] starting role=${role} specifier=${specifier ?? "none"} model=${model?.id ?? "none"}`);

    if (!model) {
        throw new Error("[graphify proxy] no model available — configure a model in OMP settings");
    }

    let activeCount = 0;
    let chunkCount = 0;
    const queue: Array<() => void> = [];

    function acquireSlot(): Promise<void> {
        if (activeCount < MAX_CONCURRENT) { activeCount++; return Promise.resolve(); }
        return new Promise(resolve => queue.push(resolve));
    }

    function releaseSlot(): void {
        activeCount--;
        if (queue.length > 0) { activeCount++; queue.shift()!(); }
    }

    // Single raw LLM call — completeSimple bypasses OMP's agent loop entirely.
    // graphify sends: system=_EXTRACTION_SYSTEM, messages=[{role:user,content:files}]
    // Response must be pure JSON — _parse_llm_json returns empty nodes on failure.
    async function callLLM(system: string, userText: string, maxTokens: number): Promise<string> {
        const result = await completeSimple(model, {
            systemPrompt: system ? [system] : undefined,
            messages: [{ role: "user", content: [{ type: "text", text: userText }], timestamp: Date.now() }],
        }, { maxTokens });
        return (result.content ?? [])
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text ?? "")
            .join("")
            .trim();
    }

    const server = Bun.serve({
        port: 0,
        async fetch(req: Request) {
            const url = new URL(req.url);
            if (req.method !== "POST" || !url.pathname.endsWith("/messages")) {
                return new Response("not found", { status: 404 });
            }
            try {
                const body = await req.json() as any;

                // Build system prompt — deepMode adds aggressive INFERRED instructions.
                const system: string = buildExtractionSystem(deepMode);

                const messages: any[] = body.messages ?? [];
                const lastUser = [...messages].reverse().find((m: any) => m.role === "user");
                const userText: string = Array.isArray(lastUser?.content)
                    ? lastUser.content.map((b: any) => (typeof b === "string" ? b : (b.text ?? ""))).join("\n")
                    : typeof lastUser?.content === "string" ? lastUser.content : "";

                const maxTokens: number = body.max_tokens ?? 4096;

                await acquireSlot();
                const n = ++chunkCount;
                logger.debug(`[DBG proxy] chunk ${n} active=${activeCount} systemLen=${system.length} msgLen=${userText.length}`);

                let text = "";
                try {
                    text = await callLLM(system, userText, maxTokens);
                    logger.debug(`[DBG proxy] chunk ${n} done responseLen=${text.length} preview=${JSON.stringify(text.slice(0, 120))}`);
                } finally {
                    releaseSlot();
                }

                return new Response(JSON.stringify({
                    id: `msg_omp_${n}`,
                    type: "message",
                    role: "assistant",
                    content: [{ type: "text", text }],
                    model: body.model ?? "claude-omp-proxy",
                    stop_reason: "end_turn",
                    stop_sequence: null,
                    usage: { input_tokens: 0, output_tokens: 0 },
                }), { headers: { "Content-Type": "application/json" } });

            } catch (err) {
                logger.debug(`[DBG proxy] error: ${String(err)}`);
                return new Response(JSON.stringify({
                    type: "error",
                    error: { type: "api_error", message: String(err) },
                }), { status: 500, headers: { "Content-Type": "application/json" } });
            }
        },
    });

    logger.debug(`[DBG proxy] server ready port=${server.port}`);
    return {
        port: server.port,
        stop: () => { server.stop(true); return chunkCount; },
    };
}

// ── subprocess helper ─────────────────────────────────────────────────────────

async function runGraphify(pi: any, argv: string[], ctx: any, hasBackend: boolean): Promise<{ output: string; chunkCount: number }> {
    const logger = pi.pi.logger;
    const env = { ...process.env } as Record<string, string>;
    delete env.GEMINI_API_KEY;
    delete env.GOOGLE_API_KEY;

    const isExtractionCmd = argv[0] === "extract";
    // --mode deep is a skill-level flag, not a graphify CLI flag — strip before spawn.
    const deepMode = argv.includes("--mode") && argv[argv.indexOf("--mode") + 1] === "deep"
        || argv.includes("--mode=deep");
    const spawnBase = argv.filter((a, i) =>
        !(a === "--mode" && argv[i + 1] === "deep") &&
        !(a === "deep" && argv[i - 1] === "--mode") &&
        a !== "--mode=deep"
    );
    let proxy: { port: number; stop: () => number } | null = null;

    if (isExtractionCmd && !hasBackend) {
        if (!ctx.model) {
            logger.debug("[DBG graphify] no model on ctx — running AST only");
        } else {
            try {
                proxy = await anthropicProxyServer(pi, ctx, deepMode);
                if (proxy) {
                    env.ANTHROPIC_BASE_URL = `http://localhost:${proxy.port}`;
                    env.ANTHROPIC_API_KEY = "omp-internal";
                    logger.debug(`[DBG graphify] proxy active port=${proxy.port}`);
                } else {
                    logger.debug("[DBG graphify] proxy not available — extract runs without OMP LLM routing");
                }
            } catch (err) {
                logger.debug(`[DBG graphify] proxy start failed: ${String(err)} — running AST only`);
            }
        }
    }

    // Smaller token budget reduces chunk size → avoids adaptive-retry bisection.
    const spawnArgv = (proxy && !spawnBase.includes("--token-budget"))
        ? [...spawnBase, "--token-budget", "20000"]
        : spawnBase;
    const python = detectPython();
    logger.debug(`[DBG graphify] python=${python} spawn argv=${JSON.stringify(spawnArgv)} hasProxy=${!!proxy} hasBackend=${hasBackend} deepMode=${deepMode}`);

    const proc = Bun.spawn([python, "-m", "graphify", ...spawnArgv], {
        cwd: process.cwd(),
        env,
        stdout: "pipe",
        stderr: "pipe",
    });

    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);

    const chunkCount = proxy?.stop() ?? 0;
    logger.debug(`[DBG graphify] exit=${exitCode} chunks=${chunkCount} stdoutLen=${stdout.trim().length} stderrLen=${stderr.trim().length}`);

    const parts: string[] = [];
    if (stdout.trim()) parts.push(stdout.trimEnd());
    if (exitCode !== 0 && stderr.trim()) parts.push(stderr.trimEnd());
    return { output: parts.join("\n"), chunkCount };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function graphExists(): boolean {
    return existsSync(join(process.cwd(), graphifyOut(), "graph.json"));
}

function readGraphReport(): string {
    try {
        return readFileSync(join(process.cwd(), graphifyOut(), "GRAPH_REPORT.md"), "utf-8");
    } catch {
        return "";
    }
}

function summarizeReport(report: string): string {
    const m = report.match(/^-\s+(\d+)\s+nodes\s*·\s*\d+\s+edges\s*·\s*(\d+)\s+communities/m);
    return m ? `${m[1]} nodes, ${m[2]} communities` : "";
}

function isSearchOrFind(event: any): boolean {
    const name = event.toolName;
    if (name === "search" || name === "find") return true;
    if (name === "bash") {
        const cmd: string = (event.input as { command?: string }).command ?? "";
        return /grep|rg|ripgrep|find |fd /.test(cmd);
    }
    return false;
}


// ── community labeling (Step 5 from graphify skill) ──────────────────────────

// Reads analysis + graph.json, returns "cid: label1, label2, ..." lines for LLM.
const COMMUNITY_SAMPLES_PY = `
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
out = Path(os.environ.get("GRAPHIFY_OUT", "graphify-out"))
analysis_path = out / ".graphify_analysis.json"
graph_path = out / "graph.json"
if not analysis_path.exists() or not graph_path.exists():
    sys.exit(0)
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
node_labels = {n["id"]: n.get("label", n["id"]) for n in graph_data.get("nodes", [])}
communities = {int(k): v for k, v in analysis["communities"].items()}
lines = []
for cid in sorted(communities):
    nodes = communities[cid][:10]
    sample = [node_labels.get(n, n) for n in nodes]
    lines.append(f"{cid}: {', '.join(sample[:8])}")
print("\\n".join(lines))
`;

async function labelCommunities(pi: any, ctx: any, targetPath: string): Promise<void> {
    const piModule = pi.pi;
    const logger = piModule.logger;
    const completeSimple = piModule.completeSimple;
    if (!completeSimple || !ctx.model) return;

    // Write script to temp file — avoids Windows arg-length limits with -c
    const tmpScript = require("node:os").tmpdir() + "/graphify_samples.py";
    require("node:fs").writeFileSync(tmpScript, COMMUNITY_SAMPLES_PY, "utf-8");
    const sampleProc = Bun.spawnSync([detectPython(), tmpScript], { cwd: process.cwd() });
    const sampleStderr = new TextDecoder("utf-8").decode(sampleProc.stderr).trim();
    const samples = new TextDecoder("utf-8").decode(sampleProc.stdout).trim();
    logger.debug(`[DBG graphify] labelCommunities: sampleProc exit=${sampleProc.exitCode} stdoutLen=${samples.length} stderr=${sampleStderr.slice(0,200)}`);
    if (!samples) { logger.debug("[DBG graphify] labelCommunities: no communities found"); return; }

    // Load existing labels — only re-label community IDs not already present.
    const labelsPath = join(process.cwd(), graphifyOut(), ".graphify_labels.json");
    let existingLabels: Record<string, string> = {};
    try {
        existingLabels = JSON.parse(require("node:fs").readFileSync(labelsPath, "utf-8"));
    } catch { /* no existing labels — label everything */ }

    const allLines = samples.split("\n");
    const toLabel = allLines.filter(l => !existingLabels[l.split(":")[0].trim()]);
    logger.debug(`[DBG graphify] labelCommunities: ${allLines.length} total, ${toLabel.length} need labeling (${allLines.length - toLabel.length} already labeled)`);

    if (toLabel.length === 0) {
        logger.debug("[DBG graphify] labelCommunities: all communities already labeled — skipping LLM calls");
        return;
    }

    // Batch communities — 80 per call to stay within output token budget.
    const BATCH = 80;
    const sampleLines = toLabel;
    const allLabels: Record<string, string> = { ...existingLabels };

    for (let i = 0; i < sampleLines.length; i += BATCH) {
        const batch = sampleLines.slice(i, i + BATCH).join("\n");
        const batchNum = Math.floor(i / BATCH) + 1;
        const totalBatches = Math.ceil(sampleLines.length / BATCH);
        logger.debug(`[DBG graphify] labelCommunities: batch ${batchNum}/${totalBatches} (${sampleLines.slice(i, i+BATCH).length} communities)`);

        const result = await completeSimple(ctx.model, {
            systemPrompt: ["Name each community in 2-5 descriptive words based on its members. Output ONLY valid JSON: {\"0\": \"Name\", \"1\": \"Name\", ...}. No explanation, no markdown fences."],
            messages: [{ role: "user", content: [{ type: "text", text: `Communities to name:\n${batch}` }], timestamp: Date.now() }],
        }, { maxTokens: 4096 });

        const raw = (result.content ?? []).filter((c: any) => c.type === "text").map((c: any) => c.text ?? "").join("").trim();
        logger.debug(`[DBG graphify] labelCommunities batch ${batchNum} response preview=${JSON.stringify(raw.slice(0, 120))}`);

        try {
            const stripped = raw.startsWith("```") ? raw.split("```")[1].replace(/^json/, "").trim() : raw;
            const parsed: Record<string, string> = JSON.parse(stripped);
            Object.assign(allLabels, parsed);
        } catch (err) {
            logger.debug(`[DBG graphify] labelCommunities batch ${batchNum} parse failed: ${String(err)}`);
        }
    }

    const labels = allLabels;
    if (Object.keys(labels).length === 0) {
        logger.debug("[DBG graphify] labelCommunities: all batches failed to parse, skipping");
        return;
    }

    require("node:fs").writeFileSync(labelsPath, JSON.stringify(allLabels), "utf-8");
    logger.debug(`[DBG graphify] labelCommunities: saved ${Object.keys(allLabels).length} labels (${toLabel.length} new)`);

    // Re-run cluster-only so GRAPH_REPORT.md picks up the labels
    const clusterProc2 = Bun.spawnSync([detectPython(), "-m", "graphify", "cluster-only", targetPath], { cwd: process.cwd() });
    const clusterStderr2 = new TextDecoder("utf-8").decode(clusterProc2.stderr).trim();
    logger.debug(`[DBG graphify] labelCommunities: cluster-only re-run exit=${clusterProc2.exitCode}${clusterStderr2 ? " stderr=" + clusterStderr2.slice(0, 150) : ""}`);
}

// ── extension factory ─────────────────────────────────────────────────────────

export default function (pi: any): void {
    let remindedThisSession = false;

    pi.on("session_start", () => {
        remindedThisSession = false;
    });

    pi.on("tool_result", (event: any): void => {
        if (remindedThisSession) return;
        if (!isSearchOrFind(event)) return;
        if (!graphExists()) return;

        remindedThisSession = true;
        const report = readGraphReport();
        const summary = report ? summarizeReport(report) : "";
        const visibleText = summary ? `[graphify] ${summary}` : `[graphify] graph ready`;
        pi.sendMessage(
            { customType: "graphify:hint", content: [{ type: "text", text: visibleText }], display: true },
            { deliverAs: "steer" },
        );
        if (report) {
            pi.sendMessage(
                { customType: "graphify:context", content: [{ type: "text", text: `${report}\n\nThe above graph report has been loaded into context. Do not narrate this. Do not output anything. Wait for the user's question.` }], display: false },
                { deliverAs: "steer" },
            );
        }
    });

    pi.registerCommand("graphify", {
        description:
            "Knowledge graph for this project. Usage: /graphify [query|path|explain|extract|update|...] [args]",
        getArgumentCompletions(prefix: string): AutocompleteItem[] {
            const TOP = [
                "query", "path", "explain",
                "extract", "update", "cluster-only",
                "add", "watch", "check-update",
                "export", "global", "prs",
                "clone", "merge-graphs",
                "hook", "tree", "dedup", "--help", "--version",
            ];
            return TOP.filter((s) => s.startsWith(prefix)).map((s) => ({ label: s, value: s }));
        },
        handler: async (args: string, ctx: any): Promise<void> => {
            const logger = pi.pi.logger;
            // update → extract: extract is fully incremental, same cost, always complete.
            // Commands that take a path default to "." if none given.
            const argv = (() => {
                const raw = args.trim() ? shellSplit(args.trim()) : [];
                const cmd = raw[0] === "update" ? "extract" : raw[0];
                const rest = raw[0] === "update" ? raw.slice(1) : raw.slice(1);
                const needsPath = ["extract", "update", "cluster-only", "query", "path", "explain"].includes(cmd);
                const hasPath = rest.length > 0 && !rest[0].startsWith("-");
                const finalRest = needsPath && !hasPath ? [".", ...rest] : rest;
                return cmd ? [cmd, ...finalRest] : raw;
            })();
            const hasBackend = argv.some((a: string) => a === "--backend" || a.startsWith("--backend="));

            logger.debug(`[DBG ext-cmd-graphify] argv=${JSON.stringify(argv)} hasBackend=${hasBackend} hasModel=${!!ctx.model}`);

            let out = "";
            let chunkCount = 0;
            try {
                ({ output: out, chunkCount } = await runGraphify(pi, argv, ctx, hasBackend));
                // add: chain extract so the new file is immediately indexed.
                if (argv[0] === "add" && !out.startsWith("error")) {
                    logger.debug("[DBG graphify] add complete — chaining extract .");
                    const extractResult = await runGraphify(pi, ["extract", "."], ctx, hasBackend);
                    chunkCount += extractResult.chunkCount;
                    out = extractResult.output || out;
                }
            } catch (err) {
                out = String(err);
            }

            // extract/add: run cluster-only, label, re-run cluster-only.
            // cluster-only: label and re-run cluster-only.
            if (argv[0] === "extract" || argv[0] === "add") {
                const proc = Bun.spawnSync([detectPython(), "-m", "graphify", "cluster-only", argv[1] ?? "."], { cwd: process.cwd() });
                const clusterStderr = new TextDecoder("utf-8").decode(proc.stderr).trim();
                logger.debug(`[DBG graphify] cluster-only exit=${proc.exitCode}${clusterStderr ? " stderr=" + clusterStderr.slice(0, 150) : ""}`);
                await labelCommunities(pi, ctx, argv[1] ?? ".");
            } else if (argv[0] === "cluster-only") {
                await labelCommunities(pi, ctx, argv[1] ?? ".");
            }

            const isExtractionCmd = argv[0] === "extract" || argv[0] === "update" || argv[0] === "add";
            if (isExtractionCmd) {
                const report = readGraphReport();
                if (report) {
                    const summary = summarizeReport(report);
                    if (summary) {
                        const chunks = ` · ${chunkCount} chunk${chunkCount !== 1 ? "s" : ""} extracted`;
                        pi.sendMessage(
                            { customType: "graphify:hint", content: [{ type: "text", text: `[graphify] ${summary}${chunks}` }], display: true },
                            { deliverAs: "steer" },
                        );
                    }
                    pi.sendMessage(
                        { customType: "graphify:context", content: [{ type: "text", text:
                            `${report}\n\nGraph updated. Do not narrate this. Do not output anything. Wait for the user's question.` }], display: false },
                        { deliverAs: "steer" },
                    );
                }
            } else if (out.trim()) {
                pi.sendMessage(
                    { customType: "graphify:output", content: [{ type: "text", text: out }], display: true },
                    { deliverAs: "steer" },
                );
            }
        },
    });
}
