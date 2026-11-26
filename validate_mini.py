"""Exercise the upstream mini CLI with scripted model responses and a local test environment."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from agenteve import write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    with tempfile.TemporaryDirectory(dir=args.out) as directory:
        with patch.dict(os.environ, {"MSWEA_GLOBAL_CONFIG_DIR": directory}), \
                create_app_session(input=DummyInput(), output=DummyOutput()):
            report = run_cases(Path(directory))
    write_json(args.out / "validation.json", report)
    print(f"mini-SWE-agent {report['version']}: {len(report['cases'])} CLI scenarios passed")


def run_cases(directory):
    from minisweagent.agents import interactive
    from minisweagent.agents.interactive import InteractiveAgent
    from minisweagent.exceptions import FormatError, Submitted
    from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
    from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions
    from minisweagent.run import mini
    model_settings = mini.get_config_from_spec(str(mini.builtin_config_dir / "mini.yaml"))["model"]

    class Environment:
        def __init__(self, events):
            self.events = events

        def execute(self, action):
            command = action["command"]
            self.events.append(f"action:{command}")
            if command == "submit":
                raise Submitted({"role": "exit", "extra": {"exit_status": "Submitted", "submission": "5"}})
            if command != "calculate":
                raise AssertionError(f"unexpected test action {command!r}")
            return {"output": "5", "returncode": 0, "exception_info": ""}

        def get_template_vars(self):
            return platform.uname()._asdict()

        def serialize(self):
            return {}

    class Model(DeterministicToolcallModel):
        def __init__(self, events, outputs):
            super().__init__(outputs=outputs, cost_per_call=0)
            self.events = events

        def query(self, messages, **kwargs):
            self.events.append("model")
            return super().query(messages, **kwargs)

        def serialize(self):
            return {"info": {"config": {"model": {"model_name": "scripted"}}}}

    def message(command):
        calls = [{"id": f"call-{command}", "type": "function",
                  "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]
        typed_calls = [SimpleNamespace(id=item["id"], function=SimpleNamespace(**item["function"]))
                       for item in calls]
        actions = parse_toolcall_actions(typed_calls, format_error_template=model_settings["format_error_template"])
        response = make_toolcall_output("", calls, actions)
        response["extra"]["cost"] = 0
        return response

    def format_error():
        try:
            parse_toolcall_actions([], format_error_template=model_settings["format_error_template"])
        except FormatError as error:
            error.messages[0]["extra"]["cost"] = 0
            return make_toolcall_output("", [], [{"raise": error}])
        raise AssertionError("empty tool calls did not produce FormatError")

    def scenario(name, outputs, expected_exit, expected_events, *, prompt_answers=(),
                 no_confirm=True, step_limit=500, time_limit=0):
        events = []
        answers = iter(prompt_answers)
        trajectory = directory / f"{name}.json"
        model = Model(events, outputs)
        environment = Environment(events)

        def prompt(_agent, _prompt, **_kwargs):
            events.append("prompt")
            return next(answers)

        overrides = [str(mini.builtin_config_dir / "mini.yaml"),
                     f"agent.step_limit={step_limit}", "agent.cost_limit=0",
                     f"agent.wall_time_limit_seconds={time_limit}"]
        def invoke():
            return mini.main(model_name=None, model_class=None, agent_class=None, environment_class=None,
                             task="Compute 2 + 3", yolo=no_confirm, cost_limit=None,
                             config_spec=overrides, output=trajectory, exit_immediately=no_confirm)

        with patch.object(mini, "configure_if_first_time"), patch.object(mini, "get_model", return_value=model), \
                patch.object(mini, "get_environment", return_value=environment), \
                patch.object(InteractiveAgent, "_stdin_is_interactive", return_value=False), \
                patch.object(InteractiveAgent, "_prompt_and_handle_slash_commands", prompt), \
                patch.object(mini, "console"), patch.object(interactive, "console"):
            model.config.observation_template = model_settings["observation_template"]
            if time_limit:
                # Only the constructor clock is offset through an inherited method hook.
                original_run = InteractiveAgent.run

                def start_at_expired_time(agent, *args, **kwargs):
                    agent._start_time -= time_limit + 1
                    return original_run(agent, *args, **kwargs)

                with patch.object(InteractiveAgent, "run", start_at_expired_time):
                    agent = invoke()
            else:
                agent = invoke()

        saved = json.loads(trajectory.read_text(encoding="utf-8"))
        assert isinstance(agent, InteractiveAgent)
        assert saved["info"]["exit_status"] == expected_exit, (name, saved["info"]["exit_status"])
        assert events == expected_events, (name, events, expected_events)
        tool_observed = any(m["role"] == "tool" and m.get("tool_call_id") == "call-calculate"
                            and m["extra"]["returncode"] == 0
                            for m in saved["messages"])
        assert tool_observed == ("action:calculate" in events), (name, tool_observed)
        return {"case": name, "agent_class": saved["info"]["config"]["agent_type"],
                "settings": {key: saved["info"]["config"]["agent"][key] for key in
                             ("mode", "confirm_exit", "step_limit", "wall_time_limit_seconds")},
                "exit_status": expected_exit, "events": events, "tool_observation_paired": tool_observed,
                "message_roles": [m["role"] for m in saved["messages"]]}

    cases = [
        scenario("tool_then_submit", [message("calculate"), message("submit")], "Submitted",
                 ["model", "action:calculate", "model", "action:submit"]),
        scenario("step_limit_without_tty", [message("calculate")], "LimitsExceeded",
                 ["model", "action:calculate"], step_limit=1),
        scenario("expired_clock", [], "TimeExceeded", [], time_limit=1),
        scenario("format_error_then_submit", [format_error(), message("submit")], "Submitted",
                 ["model", "model", "action:submit"]),
        scenario("format_error_limit", [format_error(), format_error(), format_error()],
                 "RepeatedFormatError", ["model", "model", "model"]),
        scenario("user_rejects_then_confirms", [message("calculate"), message("submit")], "Submitted",
                 ["model", "prompt", "model", "prompt", "action:submit", "prompt"],
                 prompt_answers=["skip", "", ""], no_confirm=False),
    ]
    return {"version": version("mini-swe-agent"),
            "scope": "Upstream mini.main, builtin mini.yaml, real InteractiveAgent, native bash tool-call parser/observation format and trajectory writes; scripted model and local simulated environment",
            "cases": cases}


if __name__ == "__main__":
    main()
