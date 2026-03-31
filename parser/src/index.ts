#!/usr/bin/env node
import { Command } from "commander";
import * as path from "path";
import { buildCodeGraph, writeGraphJSON } from "./graph-builder";

const program = new Command();

program
  .name("cartograph-parser")
  .description("Parse TypeScript/React/Next.js projects into code graphs")
  .version("0.1.0");

program
  .command("parse")
  .description("Parse a project directory and output a code graph JSON")
  .argument("<dir>", "Project root directory")
  .option("-o, --output <path>", "Output JSON path", "graph.json")
  .option("--snippets", "Include source code snippets in nodes", false)
  .option(
    "--max-file-size <bytes>",
    "Skip files larger than this",
    "500000"
  )
  .action(
    (
      dir: string,
      opts: { output: string; snippets: boolean; maxFileSize: string }
    ) => {
      const rootDir = path.resolve(dir);
      console.error(`[cartograph:parser] Parsing project at ${rootDir}`);

      try {
        const graph = buildCodeGraph({
          rootDir,
          includeSourceSnippets: opts.snippets,
          maxFileSize: parseInt(opts.maxFileSize, 10),
        });

        const outputPath = path.resolve(opts.output);
        writeGraphJSON(graph, outputPath);

        // summary to stdout
        const summary = {
          project: graph.projectName,
          files: graph.fileCount,
          nodes: graph.nodes.length,
          edges: graph.edges.length,
          errors: graph.parseErrors.length,
          nodeKinds: {} as Record<string, number>,
          edgeKinds: {} as Record<string, number>,
        };
        for (const n of graph.nodes) {
          summary.nodeKinds[n.kind] = (summary.nodeKinds[n.kind] || 0) + 1;
        }
        for (const e of graph.edges) {
          summary.edgeKinds[e.kind] = (summary.edgeKinds[e.kind] || 0) + 1;
        }

        console.log(JSON.stringify(summary, null, 2));
      } catch (err: any) {
        console.error(`[cartograph:parser] Fatal: ${err.message}`);
        process.exit(1);
      }
    }
  );

program.parse();
