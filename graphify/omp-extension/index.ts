import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

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

// Returns null if completeSimple is not available (non-corp OMP).
async function anthropicProxyServer(pi: any, ctx: any): Promise<{ port: number; stop: () => number } | null> {
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

                const system: string = Array.isArray(body.system)
                    ? body.system.map((b: any) => (typeof b === "string" ? b : (b.text ?? ""))).join("\n")
                    : typeof body.system === "string" ? body.system : "";

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
    let proxy: { port: number; stop: () => number } | null = null;

    if (isExtractionCmd && !hasBackend) {
        if (!ctx.model) {
            logger.debug("[DBG graphify] no model on ctx — running AST only");
        } else {
            try {
                proxy = await anthropicProxyServer(pi, ctx);
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
    const spawnArgv = (proxy && !argv.includes("--token-budget"))
        ? [...argv, "--token-budget", "20000"]
        : argv;
    logger.debug(`[DBG graphify] spawn argv=${JSON.stringify(spawnArgv)} hasProxy=${!!proxy} hasBackend=${hasBackend}`);

    const proc = Bun.spawn(["py.exe", "-m", "graphify", ...spawnArgv], {
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
    return existsSync(join(process.cwd(), "graphify-out", "graph.json"));
}

function readGraphReport(): string {
    try {
        return readFileSync(join(process.cwd(), "graphify-out", "GRAPH_REPORT.md"), "utf-8");
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
            const argv = args.trim() ? shellSplit(args.trim()) : [];
            const hasBackend = argv.some((a: string) => a === "--backend" || a.startsWith("--backend="));

            logger.debug(`[DBG ext-cmd-graphify] argv=${JSON.stringify(argv)} hasBackend=${hasBackend} hasModel=${!!ctx.model}`);

            let out = "";
            let chunkCount = 0;
            try {
                ({ output: out, chunkCount } = await runGraphify(pi, argv, ctx, hasBackend));
            } catch (err) {
                out = String(err);
            }

            // extract writes graph.json but NOT GRAPH_REPORT.md — regenerate it.
            if (argv[0] === "extract") {
                const proc = Bun.spawnSync(["py.exe", "-m", "graphify", "cluster-only", argv[1] ?? "."], { cwd: process.cwd() });
                logger.debug(`[DBG graphify] cluster-only exit=${proc.exitCode}`);
            }

            const isExtractionCmd = argv[0] === "extract" || argv[0] === "update";
            if (isExtractionCmd) {
                const report = readGraphReport();
                if (report) {
                    const summary = summarizeReport(report);
                    if (summary) {
                        const isExtract = argv[0] === "extract";
                        const chunks = isExtract
                            ? ` · ${chunkCount} chunk${chunkCount !== 1 ? "s" : ""} extracted`
                            : chunkCount > 0 ? ` · ${chunkCount} chunk${chunkCount !== 1 ? "s" : ""} extracted` : "";
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
