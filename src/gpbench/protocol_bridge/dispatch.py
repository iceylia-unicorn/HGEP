from __future__ import annotations

from gpbench.protocol_bridge.reference import (
    run_hgmp_reference_pretrain,
    run_hgprompt_reference_pretrain,
)
from gpbench.protocol_bridge.hgmp_typepair import (
    pretrain_typepair_legacy,
)


def run_protocol_pretrain(method: str, args):
    method = method.lower()

    if method == "hgmp":
        return run_hgmp_reference_pretrain(args)

    if method == "hgprompt":
        return run_hgprompt_reference_pretrain(args)

    if method == "typepair":
        return pretrain_typepair_legacy(args)

    raise ValueError(f"Unsupported method: {method}")