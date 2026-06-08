import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _function_args(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [arg.arg for arg in node.args.args]
    raise AssertionError(f"function {name} was not found")


def test_trainer_accepts_short_old_model_for_cstkr():
    tree = _module("reid/trainer.py")
    train_args = _function_args(tree, "train")
    assert "short_old_model" in train_args

    source = (ROOT / "reid/trainer.py").read_text(encoding="utf-8")
    assert "cal_CSTKR_KL" in source
    assert "build_cstkr_target" in source


def test_continual_train_preserves_short_term_model_before_fusion():
    tree = _module("continual_train.py")
    train_dataset_args = _function_args(tree, "train_dataset")
    assert "short_old_model" in train_dataset_args

    source = (ROOT / "continual_train.py").read_text(encoding="utf-8")
    assert "short_old_model = None" in source
    assert "short_old_model_candidate = copy.deepcopy(model)" in source
    assert "short_old_model=short_old_model" in source
    assert "short_old_model = short_old_model_candidate" in source
