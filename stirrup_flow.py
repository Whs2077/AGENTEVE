"""Follow Stirrup's Agent.run through its model-generating instance methods."""

import ast

from agenteve import Unsupported, dotted, function_nodes
from llm_flow import Binding, CallFlow, Cursor, Deferred, Handler, parameter_names


def class_named(source, name):
    for node in source.tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise Unsupported(f"Missing class {name} in {source.path.name}")


def class_methods(owner):
    return {stmt.name: stmt for stmt in owner.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))}


class StirrupFlow(CallFlow):
    def __init__(self, source, protocol):
        super().__init__(source)
        self.protocol_source = protocol
        contract = class_named(protocol, "LLMClient")
        self.generate = class_methods(contract).get("generate")
        if not isinstance(self.generate, ast.AsyncFunctionDef):
            raise Unsupported("Stirrup LLMClient.generate must be an async protocol method")
        self.owner = class_named(source, "Agent")
        self.methods = class_methods(self.owner)
        initializer = self.methods.get("__init__")
        if initializer is None or "run" not in self.methods:
            raise Unsupported("Stirrup requires Agent.__init__ and Agent.run")
        client_types = {
            alias.asname or alias.name
            for stmt in source.tree.body
            if isinstance(stmt, ast.ImportFrom)
            and (stmt.module, stmt.level) in {("stirrup.core.models", 0), ("models", 1)}
            for alias in stmt.names if alias.name == "LLMClient"
        }
        self.client_fields = self.find_client_fields(initializer, client_types)
        direct_calls = {name: {dotted(node.func) for node in function_nodes(fn) if isinstance(node, ast.Call)}
                        for name, fn in self.methods.items()}
        self.model_methods = {name for name, calls in direct_calls.items()
                              if calls & {f"{field}.generate" for field in self.client_fields}}
        while True:
            forward = {f"self.{name}" for name in self.model_methods}
            expanded = self.model_methods | {name for name, calls in direct_calls.items() if calls & forward}
            if expanded == self.model_methods:
                break
            self.model_methods = expanded
        if "run" not in self.model_methods:
            raise Unsupported("No typed LLMClient.generate call reachable from Agent.run")

    @staticmethod
    def find_client_fields(initializer, client_types):
        parameters = (*initializer.args.posonlyargs, *initializer.args.args, *initializer.args.kwonlyargs)
        client_names = {p.arg for p in parameters if dotted(p.annotation) in client_types}
        fields = {}
        for node in function_nodes(initializer):
            if (isinstance(node, ast.AnnAssign)
                    and dotted(node.annotation) in client_types
                    and isinstance(node.value, ast.Name) and node.value.id in client_names):
                field = dotted(node.target)
                if field.startswith("self.") and field.count(".") == 1:
                    fields[field] = node
        return fields

    def expression(self, expr, cursor, handler):
        value = super().expression(expr, cursor, handler)
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
            return value
        receiver = dotted(expr.func.value)
        name = expr.func.attr
        if name == "generate" and receiver in self.client_fields:
            binding = Binding("model", f"{receiver}.model_slug", self.client_fields[receiver])
            return Binding("deferred", Deferred(binding, expr))
        if receiver == "self" and name in self.model_methods:
            deferred = Deferred(Binding("method", name), expr)
            if isinstance(self.methods[name], ast.AsyncFunctionDef):
                return Binding("deferred", deferred)
            self.invoke(deferred, cursor, handler)
        return value

    def invoke(self, deferred, cursor, handler, wait=None):
        if deferred.receiver.kind != "method":
            super().invoke(deferred, cursor, handler, wait)
            return
        name = deferred.receiver.value
        if name in handler.builders:
            raise Unsupported(f"Recursive Stirrup agent method: {name}")
        self.advance(cursor, deferred.call, step=f"call Agent.{name}")
        returned = self.vertex()
        self.inline(name, cursor.position, returned, handler.builders)
        cursor.position = returned
        self.advance(cursor, deferred.call, step=f"return from Agent.{name}")

    def inline(self, name, entry, finish, stack):
        function = self.methods[name]
        env = self.globals.copy()
        local_names = parameter_names(function.args)
        local_names.update(node.id for node in function_nodes(function)
                           if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)))
        self.assign(env, local_names, None)
        handler = Handler(f"Agent.{name}", finish, {}, set(), False, (*stack, name))
        for cursor in self.block(function.body, [Cursor(entry, env)], handler):
            self.link(cursor.position, finish, self.source.evidence(function))

    def extract(self):
        entry, end = self.vertex(), self.vertex()
        self.inline("run", entry, end, ())
        anchor = self.source.evidence(self.methods["run"])
        result = self.project(
            entry, end, "Agent.run", {"__start__": [anchor], "__end__": [anchor]},
            extractor="stirrup-llm",
            analysis_scope=(
                "Static normal paths through Agent.run and same-class methods that reach typed LLMClient.generate. "
                "Branch guards are structural witnesses; history-dependent constraints are not solved. "
                "Exception handlers, client retries and tool-embedded agents are outside scope."
            ),
        )
        result["sources"].append(self.protocol_source.metadata())
        evidence = self.protocol_source.evidence(self.generate)
        for node in result["nodes"]:
            if node["kind"] == "llm_call":
                node["evidence"].append(evidence)
        return result


def extract_stirrup(source, protocol):
    return StirrupFlow(source, protocol).extract()
