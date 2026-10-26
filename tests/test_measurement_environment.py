from __future__ import annotations

import pytest

from adaptivecarearena import RecordedQuestionnaireEnv


class AnswerConditionedPolicy:
    def reset(self, items, budget):
        self.started = False

    def choose(self, revealed, available, remaining_budget):
        if not self.started:
            self.started = True
            return "item_a"
        if revealed["item_a"] > 0:
            return "item_b" if "item_b" in available else None
        return None


def test_sequential_replay_reveals_only_chosen_recorded_answers():
    env = RecordedQuestionnaireEnv(
        {"item_a": 2.0, "item_b": 1.0, "item_c": 3.0},
        budget=2,
    )
    transitions = env.run(AnswerConditionedPolicy())
    assert [step.revealed_item for step in transitions] == ["item_a", "item_b"]
    assert transitions[-1].reason == "budget_exhausted"
    assert dict(transitions[-1].observation.revealed) == {"item_a": 2.0, "item_b": 1.0}
    assert "item_c" not in transitions[-1].observation.revealed


def test_policy_can_stop_before_budget_is_spent():
    env = RecordedQuestionnaireEnv({"item_a": 0.0, "item_b": 2.0}, budget=2)
    transitions = env.run(AnswerConditionedPolicy())
    assert [step.revealed_item for step in transitions] == ["item_a", None]
    assert transitions[-1].reason == "policy_stop"
    assert transitions[-1].observation.spent == 1.0


def test_environment_rejects_hidden_duplicate_and_over_budget_actions():
    env = RecordedQuestionnaireEnv(
        {"item_a": 1.0, "item_b": 2.0}, budget=1.0, costs={"item_a": 1.0, "item_b": 2.0}
    )
    observation = env.reset()
    assert observation.available == ("item_a",)
    with pytest.raises(ValueError, match="not currently available"):
        env.step("item_b")
    transition = env.step("item_a")
    assert transition.reason == "budget_exhausted"
    with pytest.raises(RuntimeError, match="terminated"):
        env.step("item_a")
