/**
 * Core types for Cartograph's code graph representation.
 *
 * A codebase is modeled as a directed heterogeneous graph:
 *   - Nodes are code entities (functions, components, variables, classes, types)
 *   - Edges are relationships (calls, imports, data_flow, jsx_renders, type_refs)
 *
 * This representation is consumed by the ML engine for module discovery (GraphSAGE)
 * and change impact prediction (GAT).
 */

export type NodeKind =
  | "function"
  | "arrow_function"
  | "component"       // React component (detected via JSX return + PascalCase)
  | "hook"            // React hook (use* naming convention)
  | "class"
  | "variable"
  | "type_alias"
  | "interface"
  | "enum"
  | "api_route"       // Next.js API route handler
  | "page"            // Next.js page component
  | "middleware";      // Next.js middleware

export type EdgeKind =
  | "calls"           // function A invokes function B
  | "imports"         // file A imports from file B
  | "data_flow"       // variable defined in A, read in B
  | "jsx_renders"     // component A renders component B in JSX
  | "type_ref"        // A references type defined in B
  | "prop_passes"     // parent passes prop to child component
  | "state_reads"     // component reads from a shared state/context
  | "state_writes";   // component writes to a shared state/context

export interface CodeNode {
  id: string;                     // deterministic: sha256(filePath + name + kind)
  name: string;
  kind: NodeKind;
  filePath: string;               // relative to project root
  startLine: number;
  endLine: number;
  exported: boolean;
  async: boolean;
  features: NodeFeatures;
  sourceSnippet?: string;         // first 500 chars of source, for LLM context
}

export interface NodeFeatures {
  loc: number;                    // lines of code
  cyclomaticComplexity: number;   // branches + 1
  paramCount: number;
  returnCount: number;            // number of return statements
  calleeCount: number;            // how many other functions this calls
  callerCount: number;            // how many functions call this (populated after graph build)
  depthInFileTree: number;        // path depth from project root
  hasJSX: boolean;
  hasTryCatch: boolean;
  hasAwait: boolean;
  importCount: number;            // number of imports in the file
  jsxChildCount: number;          // number of JSX children rendered
  propsCount: number;             // number of props accepted (for components)
}

export interface CodeEdge {
  id: string;
  sourceId: string;
  targetId: string;
  kind: EdgeKind;
  weight: number;                 // 1.0 default; higher = stronger coupling
  metadata?: Record<string, unknown>;
}

export interface CodeGraph {
  projectName: string;
  rootDir: string;
  nodes: CodeNode[];
  edges: CodeEdge[];
  fileCount: number;
  parseErrors: ParseError[];
  parsedAt: string;               // ISO timestamp
}

export interface ParseError {
  filePath: string;
  message: string;
  line?: number;
}

export interface ParserOptions {
  rootDir: string;
  includeSourceSnippets?: boolean;
  maxFileSize?: number;           // bytes, skip files larger than this
  ignorePatterns?: string[];
}
