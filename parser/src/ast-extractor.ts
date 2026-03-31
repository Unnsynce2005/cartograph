import {
  Project,
  SourceFile,
  SyntaxKind,
  FunctionDeclaration,
  VariableDeclaration,
  ArrowFunction,
  ClassDeclaration,
  InterfaceDeclaration,
  TypeAliasDeclaration,
  EnumDeclaration,
  Node,
  CallExpression,
  JsxElement,
  JsxSelfClosingElement,
  ImportDeclaration,
} from "ts-morph";
import * as crypto from "crypto";
import * as path from "path";
import {
  CodeNode,
  CodeEdge,
  NodeKind,
  EdgeKind,
  NodeFeatures,
  ParseError,
  ParserOptions,
} from "./types";

function nodeId(filePath: string, name: string, kind: string): string {
  const raw = `${filePath}::${name}::${kind}`;
  return crypto.createHash("sha256").update(raw).digest("hex").slice(0, 16);
}

function edgeId(srcId: string, tgtId: string, kind: string): string {
  return crypto
    .createHash("sha256")
    .update(`${srcId}->${tgtId}::${kind}`)
    .digest("hex")
    .slice(0, 16);
}

function isPascalCase(name: string): boolean {
  return /^[A-Z][a-zA-Z0-9]*$/.test(name);
}

function isHookName(name: string): boolean {
  return /^use[A-Z]/.test(name);
}

function containsJSX(node: Node): boolean {
  return (
    node.getDescendantsOfKind(SyntaxKind.JsxElement).length > 0 ||
    node.getDescendantsOfKind(SyntaxKind.JsxSelfClosingElement).length > 0 ||
    node.getDescendantsOfKind(SyntaxKind.JsxFragment).length > 0
  );
}

function countBranches(node: Node): number {
  let count = 1;
  count += node.getDescendantsOfKind(SyntaxKind.IfStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.ConditionalExpression).length;
  count += node.getDescendantsOfKind(SyntaxKind.SwitchStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.ForStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.ForInStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.ForOfStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.WhileStatement).length;
  count += node.getDescendantsOfKind(SyntaxKind.CatchClause).length;
  count += node.getDescendantsOfKind(SyntaxKind.BinaryExpression).filter(
    (b) => {
      const op = b.getOperatorToken().getText();
      return op === "&&" || op === "||" || op === "??";
    }
  ).length;
  return count;
}

function classifyFunction(
  name: string,
  node: Node,
  filePath: string,
  exported: boolean
): NodeKind {
  // Next.js API route
  if (
    filePath.includes("/api/") &&
    (name === "handler" ||
      name === "GET" ||
      name === "POST" ||
      name === "PUT" ||
      name === "DELETE" ||
      name === "PATCH")
  ) {
    return "api_route";
  }
  // Next.js page
  if (
    (name === "default" || name === "Page" || name === "Home") &&
    exported &&
    (filePath.includes("/app/") || filePath.includes("/pages/"))
  ) {
    return "page";
  }
  // Next.js middleware
  if (name === "middleware" && filePath.includes("middleware")) {
    return "middleware";
  }
  // React hook
  if (isHookName(name)) return "hook";
  // React component (PascalCase + returns JSX)
  if (isPascalCase(name) && containsJSX(node)) return "component";
  return "function";
}

function extractFeatures(
  node: Node,
  filePath: string,
  rootDir: string,
  paramCount: number
): NodeFeatures {
  const text = node.getText();
  const lines = text.split("\n").length;
  const hasJSX = containsJSX(node);
  const jsxChildren =
    node.getDescendantsOfKind(SyntaxKind.JsxElement).length +
    node.getDescendantsOfKind(SyntaxKind.JsxSelfClosingElement).length;

  // count props for components
  let propsCount = 0;
  const params = node.getDescendantsOfKind(SyntaxKind.Parameter);
  if (params.length > 0) {
    const firstParam = params[0];
    const bindings = firstParam.getDescendantsOfKind(
      SyntaxKind.BindingElement
    );
    propsCount = bindings.length || (firstParam.getType() ? 1 : 0);
  }

  return {
    loc: lines,
    cyclomaticComplexity: countBranches(node),
    paramCount,
    returnCount: node.getDescendantsOfKind(SyntaxKind.ReturnStatement).length,
    calleeCount: node.getDescendantsOfKind(SyntaxKind.CallExpression).length,
    callerCount: 0, // populated later
    depthInFileTree: path.relative(rootDir, filePath).split(path.sep).length,
    hasJSX,
    hasTryCatch:
      node.getDescendantsOfKind(SyntaxKind.TryStatement).length > 0,
    hasAwait:
      node.getDescendantsOfKind(SyntaxKind.AwaitExpression).length > 0,
    importCount: 0, // set at file level
    jsxChildCount: jsxChildren,
    propsCount,
  };
}

export class ASTExtractor {
  private project: Project;
  private options: ParserOptions;
  private nodes: Map<string, CodeNode> = new Map();
  private edges: CodeEdge[] = [];
  private errors: ParseError[] = [];
  // maps: functionName -> nodeId (for call resolution)
  private nameToId: Map<string, string[]> = new Map();
  // maps: filePath -> list of exported names
  private fileExports: Map<string, Map<string, string>> = new Map();

  constructor(options: ParserOptions) {
    this.options = options;
    this.project = new Project({
      tsConfigFilePath: path.join(options.rootDir, "tsconfig.json"),
      skipAddingFilesFromTsConfig: true,
    });
  }

  addFiles(): number {
    const ignore = this.options.ignorePatterns || [
      "**/node_modules/**",
      "**/dist/**",
      "**/build/**",
      "**/.next/**",
      "**/*.test.*",
      "**/*.spec.*",
      "**/__tests__/**",
    ];

    this.project.addSourceFilesAtPaths(
      path.join(this.options.rootDir, "**/*.{ts,tsx,js,jsx}")
    );

    // remove ignored files
    const files = this.project.getSourceFiles();
    let kept = 0;
    for (const file of files) {
      const rel = path.relative(this.options.rootDir, file.getFilePath());
      const shouldIgnore = ignore.some((pattern) => {
        const simple = pattern.replace(/\*\*/g, "").replace(/\*/g, "");
        return rel.includes(simple.replace(/\//g, path.sep));
      });
      if (shouldIgnore) {
        this.project.removeSourceFile(file);
      } else {
        kept++;
      }
    }
    return kept;
  }

  extractAll(): { nodes: CodeNode[]; edges: CodeEdge[]; errors: ParseError[] } {
    const files = this.project.getSourceFiles();

    // pass 1: extract all nodes and build name->id mapping
    for (const file of files) {
      try {
        this.extractFileNodes(file);
      } catch (e: any) {
        this.errors.push({
          filePath: path.relative(
            this.options.rootDir,
            file.getFilePath()
          ),
          message: e.message || String(e),
        });
      }
    }

    // pass 2: extract edges (calls, imports, JSX renders, data flow)
    for (const file of files) {
      try {
        this.extractFileEdges(file);
      } catch (e: any) {
        this.errors.push({
          filePath: path.relative(
            this.options.rootDir,
            file.getFilePath()
          ),
          message: `Edge extraction: ${e.message || String(e)}`,
        });
      }
    }

    // pass 3: populate callerCount
    for (const edge of this.edges) {
      if (edge.kind === "calls" || edge.kind === "jsx_renders") {
        const target = this.nodes.get(edge.targetId);
        if (target) {
          target.features.callerCount++;
        }
      }
    }

    return {
      nodes: Array.from(this.nodes.values()),
      edges: this.edges,
      errors: this.errors,
    };
  }

  private extractFileNodes(file: SourceFile): void {
    const filePath = path.relative(this.options.rootDir, file.getFilePath());
    const importCount = file.getImportDeclarations().length;
    const exportMap = new Map<string, string>();

    // functions
    for (const fn of file.getFunctions()) {
      const name = fn.getName() || "anonymous";
      const exported = fn.isExported();
      const kind = classifyFunction(name, fn, filePath, exported);
      const id = nodeId(filePath, name, kind);

      const node: CodeNode = {
        id,
        name,
        kind,
        filePath,
        startLine: fn.getStartLineNumber(),
        endLine: fn.getEndLineNumber(),
        exported,
        async: fn.isAsync(),
        features: {
          ...extractFeatures(
            fn,
            filePath,
            this.options.rootDir,
            fn.getParameters().length
          ),
          importCount,
        },
        sourceSnippet: this.options.includeSourceSnippets
          ? fn.getText().slice(0, 500)
          : undefined,
      };

      this.nodes.set(id, node);
      this.registerName(name, id);
      if (exported) exportMap.set(name, id);
    }

    // variable declarations (arrow functions, consts)
    for (const varStmt of file.getVariableStatements()) {
      for (const decl of varStmt.getDeclarations()) {
        const init = decl.getInitializer();
        if (!init) continue;

        const name = decl.getName();
        const exported = varStmt.isExported();

        if (
          init.getKind() === SyntaxKind.ArrowFunction ||
          init.getKind() === SyntaxKind.FunctionExpression
        ) {
          const kind = classifyFunction(name, init, filePath, exported);
          const id = nodeId(filePath, name, kind);
          const arrowFn = init as ArrowFunction;

          const node: CodeNode = {
            id,
            name,
            kind,
            filePath,
            startLine: decl.getStartLineNumber(),
            endLine: decl.getEndLineNumber(),
            exported,
            async: arrowFn.isAsync(),
            features: {
              ...extractFeatures(
                init,
                filePath,
                this.options.rootDir,
                arrowFn.getParameters().length
              ),
              importCount,
            },
            sourceSnippet: this.options.includeSourceSnippets
              ? decl.getText().slice(0, 500)
              : undefined,
          };

          this.nodes.set(id, node);
          this.registerName(name, id);
          if (exported) exportMap.set(name, id);
        } else {
          // regular variable
          const id = nodeId(filePath, name, "variable");
          const node: CodeNode = {
            id,
            name,
            kind: "variable",
            filePath,
            startLine: decl.getStartLineNumber(),
            endLine: decl.getEndLineNumber(),
            exported,
            async: false,
            features: {
              ...extractFeatures(init, filePath, this.options.rootDir, 0),
              importCount,
            },
          };
          this.nodes.set(id, node);
          this.registerName(name, id);
          if (exported) exportMap.set(name, id);
        }
      }
    }

    // classes
    for (const cls of file.getClasses()) {
      const name = cls.getName() || "AnonymousClass";
      const id = nodeId(filePath, name, "class");
      const exported = cls.isExported();

      const node: CodeNode = {
        id,
        name,
        kind: "class",
        filePath,
        startLine: cls.getStartLineNumber(),
        endLine: cls.getEndLineNumber(),
        exported,
        async: false,
        features: {
          ...extractFeatures(cls, filePath, this.options.rootDir, 0),
          importCount,
          loc: cls.getText().split("\n").length,
        },
      };
      this.nodes.set(id, node);
      this.registerName(name, id);
      if (exported) exportMap.set(name, id);
    }

    // interfaces
    for (const iface of file.getInterfaces()) {
      const name = iface.getName();
      const id = nodeId(filePath, name, "interface");
      this.nodes.set(id, {
        id,
        name,
        kind: "interface",
        filePath,
        startLine: iface.getStartLineNumber(),
        endLine: iface.getEndLineNumber(),
        exported: iface.isExported(),
        async: false,
        features: {
          loc: iface.getText().split("\n").length,
          cyclomaticComplexity: 1,
          paramCount: 0,
          returnCount: 0,
          calleeCount: 0,
          callerCount: 0,
          depthInFileTree: path
            .relative(this.options.rootDir, filePath)
            .split(path.sep).length,
          hasJSX: false,
          hasTryCatch: false,
          hasAwait: false,
          importCount,
          jsxChildCount: 0,
          propsCount: iface.getProperties().length,
        },
      });
      this.registerName(name, id);
      if (iface.isExported()) exportMap.set(name, id);
    }

    // type aliases
    for (const ta of file.getTypeAliases()) {
      const name = ta.getName();
      const id = nodeId(filePath, name, "type_alias");
      this.nodes.set(id, {
        id,
        name,
        kind: "type_alias",
        filePath,
        startLine: ta.getStartLineNumber(),
        endLine: ta.getEndLineNumber(),
        exported: ta.isExported(),
        async: false,
        features: {
          loc: ta.getText().split("\n").length,
          cyclomaticComplexity: 1,
          paramCount: 0,
          returnCount: 0,
          calleeCount: 0,
          callerCount: 0,
          depthInFileTree: path
            .relative(this.options.rootDir, filePath)
            .split(path.sep).length,
          hasJSX: false,
          hasTryCatch: false,
          hasAwait: false,
          importCount,
          jsxChildCount: 0,
          propsCount: 0,
        },
      });
      this.registerName(name, id);
      if (ta.isExported()) exportMap.set(name, id);
    }

    // enums
    for (const en of file.getEnums()) {
      const name = en.getName();
      const id = nodeId(filePath, name, "enum");
      this.nodes.set(id, {
        id,
        name,
        kind: "enum",
        filePath,
        startLine: en.getStartLineNumber(),
        endLine: en.getEndLineNumber(),
        exported: en.isExported(),
        async: false,
        features: {
          loc: en.getText().split("\n").length,
          cyclomaticComplexity: 1,
          paramCount: 0,
          returnCount: 0,
          calleeCount: 0,
          callerCount: 0,
          depthInFileTree: path
            .relative(this.options.rootDir, filePath)
            .split(path.sep).length,
          hasJSX: false,
          hasTryCatch: false,
          hasAwait: false,
          importCount,
          jsxChildCount: 0,
          propsCount: en.getMembers().length,
        },
      });
      this.registerName(name, id);
    }

    this.fileExports.set(filePath, exportMap);
  }

  private extractFileEdges(file: SourceFile): void {
    const filePath = path.relative(this.options.rootDir, file.getFilePath());

    // import edges
    for (const imp of file.getImportDeclarations()) {
      const moduleSpecifier = imp.getModuleSpecifierValue();
      if (moduleSpecifier.startsWith(".")) {
        // resolve relative import to a file
        const resolved = this.resolveImportPath(filePath, moduleSpecifier);
        if (!resolved) continue;

        const targetExports = this.fileExports.get(resolved);
        if (!targetExports) continue;

        // named imports
        for (const named of imp.getNamedImports()) {
          const importedName = named.getName();
          const targetId = targetExports.get(importedName);
          if (!targetId) continue;

          // find all nodes in current file that could be the importer
          const sourceNodes = this.getFileNodes(filePath);
          for (const srcNode of sourceNodes) {
            // check if this node actually uses the imported name
            const nodeAst = this.findNodeInFile(file, srcNode);
            if (nodeAst && this.nodeReferencesName(nodeAst, importedName)) {
              this.addEdge(srcNode.id, targetId, "imports");
            }
          }
        }

        // default import
        const defaultImport = imp.getDefaultImport();
        if (defaultImport) {
          const targetId =
            targetExports.get("default") ||
            targetExports.values().next().value;
          if (targetId) {
            const importName = defaultImport.getText();
            // link file-level dependency
            const srcNodes = this.getFileNodes(filePath);
            for (const srcNode of srcNodes) {
              const nodeAst = this.findNodeInFile(file, srcNode);
              if (nodeAst && this.nodeReferencesName(nodeAst, importName)) {
                this.addEdge(srcNode.id, targetId, "imports");
              }
            }
          }
        }
      }
    }

    // call edges and JSX render edges
    const callExpressions = file.getDescendantsOfKind(
      SyntaxKind.CallExpression
    );
    for (const call of callExpressions) {
      const calleeName = this.getCallExpressionName(call);
      if (!calleeName) continue;

      const callerNode = this.findEnclosingNode(filePath, call);
      if (!callerNode) continue;

      const targetIds = this.nameToId.get(calleeName);
      if (!targetIds) continue;

      for (const targetId of targetIds) {
        if (targetId !== callerNode.id) {
          this.addEdge(callerNode.id, targetId, "calls");
        }
      }
    }

    // JSX renders
    const jsxElements = [
      ...file.getDescendantsOfKind(SyntaxKind.JsxElement),
      ...file.getDescendantsOfKind(SyntaxKind.JsxSelfClosingElement),
    ];
    for (const jsx of jsxElements) {
      let tagName: string;
      if (jsx.getKind() === SyntaxKind.JsxElement) {
        const openTag = (jsx as JsxElement).getOpeningElement();
        tagName = openTag.getTagNameNode().getText();
      } else {
        tagName = (jsx as JsxSelfClosingElement).getTagNameNode().getText();
      }

      // only user-defined components (PascalCase)
      if (!isPascalCase(tagName)) continue;

      const parentNode = this.findEnclosingNode(filePath, jsx);
      if (!parentNode) continue;

      const targetIds = this.nameToId.get(tagName);
      if (!targetIds) continue;

      for (const targetId of targetIds) {
        if (targetId !== parentNode.id) {
          this.addEdge(parentNode.id, targetId, "jsx_renders");
        }
      }
    }
  }

  private registerName(name: string, id: string): void {
    const existing = this.nameToId.get(name) || [];
    existing.push(id);
    this.nameToId.set(name, existing);
  }

  private addEdge(srcId: string, tgtId: string, kind: EdgeKind): void {
    const id = edgeId(srcId, tgtId, kind);
    // deduplicate
    if (this.edges.find((e) => e.id === id)) return;
    this.edges.push({ id, sourceId: srcId, targetId: tgtId, kind, weight: 1.0 });
  }

  private getFileNodes(filePath: string): CodeNode[] {
    return Array.from(this.nodes.values()).filter(
      (n) => n.filePath === filePath
    );
  }

  private findNodeInFile(file: SourceFile, node: CodeNode): Node | undefined {
    // find the AST node corresponding to our CodeNode by line number
    const pos = file.compilerNode.getPositionOfLineAndCharacter(
      node.startLine - 1,
      0
    );
    return file.getDescendantAtPos(pos);
  }

  private nodeReferencesName(node: Node, name: string): boolean {
    const identifiers = node.getDescendantsOfKind(SyntaxKind.Identifier);
    return identifiers.some((id) => id.getText() === name);
  }

  private findEnclosingNode(
    filePath: string,
    astNode: Node
  ): CodeNode | undefined {
    const line = astNode.getStartLineNumber();
    const fileNodes = this.getFileNodes(filePath);
    // find the tightest enclosing node
    let best: CodeNode | undefined;
    let bestSize = Infinity;
    for (const n of fileNodes) {
      if (
        line >= n.startLine &&
        line <= n.endLine &&
        n.endLine - n.startLine < bestSize
      ) {
        best = n;
        bestSize = n.endLine - n.startLine;
      }
    }
    return best;
  }

  private getCallExpressionName(call: CallExpression): string | undefined {
    const expr = call.getExpression();
    const text = expr.getText();
    // handle simple calls: foo(), Bar()
    if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(text)) return text;
    // handle member access: obj.method() -> method
    const match = text.match(/\.([a-zA-Z_$][a-zA-Z0-9_$]*)$/);
    return match ? match[1] : undefined;
  }

  private resolveImportPath(
    fromFile: string,
    moduleSpecifier: string
  ): string | undefined {
    const fromDir = path.dirname(fromFile);
    const resolved = path.normalize(path.join(fromDir, moduleSpecifier));
    const extensions = ["", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"];
    for (const ext of extensions) {
      const candidate = resolved + ext;
      if (this.fileExports.has(candidate)) return candidate;
    }
    return undefined;
  }
}
