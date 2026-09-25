from llm_rl.data import PROMPT_TEMPLATE, Example, infinite_batches, parse_row


DAPO_CONTENT = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\nWhat is 6*7?\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def test_dapo_style_row():
    row = {
        "prompt": [{"role": "user", "content": DAPO_CONTENT}],
        "reward_model": {"style": "rule", "ground_truth": "42"},
        "extra_info": {"index": "9a9b6eb4"},
    }
    ex = parse_row(row, source="dapo")
    assert ex is not None
    assert ex.answer == "42"
    # DAPO's own "Answer:" format instruction contradicts the \boxed{} format our
    # reward grades, so it must not survive into the prompt.
    assert ex.problem == "What is 6*7?"
    assert "Answer:" not in ex.prompt.replace("\\boxed", "")


def test_aime_style_row():
    ex = parse_row({"problem": "Find n.", "answer": 204})
    assert ex is not None and ex.answer == "204"


def test_gsm8k_style_row():
    ex = parse_row({"question": "How many?", "answer": "steps...\n#### 18"})
    assert ex is not None and ex.answer == "18"


def test_system_turn_is_dropped():
    row = {
        "prompt": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Compute 2+2."},
        ],
        "reward_model": {"ground_truth": "4"},
    }
    ex = parse_row(row)
    assert ex is not None and ex.problem == "Compute 2+2."


def test_rows_without_answer_are_dropped():
    assert parse_row({"problem": "no answer here"}) is None
    assert parse_row({"answer": "42"}) is None


def test_prompt_ends_with_assistant_cue_and_asks_for_boxed():
    ex = Example(problem="What is 1+1?", answer="2")
    assert ex.prompt.endswith("Assistant:")
    assert "\\boxed{}" in ex.prompt
    assert "What is 1+1?" in ex.prompt
    assert PROMPT_TEMPLATE.count("{problem}") == 1


def test_infinite_batches_cycles_and_is_deterministic():
    examples = [Example(problem=str(i), answer=str(i)) for i in range(5)]
    a = [b for b, _ in zip(infinite_batches(examples, 3, seed=0), range(4))]
    b = [b for b, _ in zip(infinite_batches(examples, 3, seed=0), range(4))]
    assert [[e.problem for e in batch] for batch in a] == [[e.problem for e in batch] for batch in b]
    assert all(len(batch) == 3 for batch in a)
    # Every example is seen within the first ceil(5/3) batches of an epoch.
    assert {e.problem for e in a[0] + a[1]} == {str(i) for i in range(5)}
