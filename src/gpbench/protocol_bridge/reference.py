from protocols.hgmp.run_legacy import pretrain as hgmp_reference_pretrain
from protocols.hgprompt.source.pretrain import run_model_DBLP as hgprompt_reference_pretrain


def run_hgmp_reference_pretrain(args):
    return hgmp_reference_pretrain(args)


def run_hgprompt_reference_pretrain(args):
    return hgprompt_reference_pretrain(args)