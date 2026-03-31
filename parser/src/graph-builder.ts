import * as path from "path";
import * as fs from "fs";
import { ASTExtractor } from "./ast-extractor";
import { CodeGraph, ParserOptions } from "./types";

export function buildCodeGraph(options: ParserOptions): CodeGraph {
  const extractor = new ASTExtractor(options);

  const fileCount = extractor.addFiles();
  if (fileCount === 0) {
    throw new Error(
      `No TypeScript/JavaScript files found in ${options.rootDir}`
    );
  }

  console.error(`[cartograph:parser] Parsing ${fileCount} files...`);

  const { nodes, edges, errors } = extractor.extractAll();

  console.error(
    `[cartograph:parser] Extracted ${nodes.length} nodes, ${edges.length} edges, ${errors.length} errors`
  );

  // compute edge statistics
  const edgesByKind: Record<string, number> = {};
  for (const edge of edges) {
    edgesByKind[edge.kind] = (edgesByKind[edge.kind] || 0) + 1;
  }
  console.error(`[cartograph:parser] Edge breakdown:`, edgesByKind);

  const projectName = path.basename(path.resolve(options.rootDir));

  return {
    projectName,
    rootDir: path.resolve(options.rootDir),
    nodes,
    edges,
    fileCount,
    parseErrors: errors,
    parsedAt: new Date().toISOString(),
  };
}

export function writeGraphJSON(graph: CodeGraph, outputPath: string): void {
  fs.writeFileSync(outputPath, JSON.stringify(graph, null, 2), "utf-8");
  console.error(`[cartograph:parser] Wrote graph to ${outputPath}`);
}
