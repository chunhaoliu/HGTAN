from argparse import Namespace

import pytest

from run import normalize_cli_aliases
from utils.config import HGTANConfig
from utils.tools import apply_cli_overrides


def test_explicit_seed_list_overrides_mode_seeds():
    config = HGTANConfig.get_config("gpu")
    args = Namespace(itr=3, seeds=[101, 202, 303], seed=None)
    apply_cli_overrides(config, args)
    assert config["run"]["seeds"] == [101, 202, 303]
    assert config["run"]["num_runs"] == 3


def test_normalize_explicit_seeds_sets_iteration_count():
    args = Namespace(num_runs=None, itr=None, epochs=None, train_epochs=None, seeds=[7, 11, 13])
    normalize_cli_aliases(args)
    assert args.itr == 3


def test_normalize_explicit_seeds_rejects_mismatched_iteration_count():
    args = Namespace(num_runs=None, itr=2, epochs=None, train_epochs=None, seeds=[7, 11, 13])
    with pytest.raises(ValueError):
        normalize_cli_aliases(args)
