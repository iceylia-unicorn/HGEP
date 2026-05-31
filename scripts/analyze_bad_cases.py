from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dgl
import torch
import torch.nn as nn

from gpbench.downstream.model import MLPHead
from gpbench.protocol_bridge import downstream_legacy as legacy_bridge
from protocols.hgmp.prompt_legacy import HeteroPrompt
from scripts.peprompt_benchmark import (
    _make_legacy_args,
    _normalize_dataset_defaults,
    _resolve_ckpt,
    _set_global_seed,
    _sync_args_with_legacy_ckpts,
    build_parser,
    load_peprompt_offline_legacy_splits,
)
from scripts.visualizer import (
    ForwardCapture,
    LegacyModelVisualizerAdapter,
    visualize_feature_trajectory,
    visualize_node_drift_heatmap,
    visualize_subgraph_aggregation,
    visualize_target_feature_delta_bars,
    visualize_target_feature_heatmap,
    _write_dict_rows_csv,
)


@dataclass
class CaseRecord:
    case_name: str
    true_label: int
    pred_label: int
    confidence: float
    inverse_indices: dict | int
    prompt_summary: dict
    score_mode: str


@dataclass
class SamplePrediction:
    graph: dgl.DGLHeteroGraph
    inverse_indices: dict | int
    true_label: int
    pred_label: int
    confidence: float
    capture: ForwardCapture


def _load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_best_ckpt_path(args) -> Path:
    method_dir = args.method
    if args.method == "peprompt":
        method_dir = f"{args.method}.{args.subgraph_type}"
    return (
        Path(args.save_dir)
        / "aligned_protocol"
        / args.dataset
        / method_dir
        / f"{args.shot}-shot"
        / f"pretrainseed{args.pretrain_seed}"
        / f"splitseed{args.split_seed}"
        / f"repeat{args.repeat_id}"
        / "best.pt"
    )


def _build_linear_or_mlp_head(state_dict: dict, device: torch.device) -> nn.Module:
    """
    自动根据 `best.pt` 里的 state_dict 还原分类头。
    """

    if "fc1.weight" in state_dict and "fc2.weight" in state_dict:
        in_dim = int(state_dict["fc1.weight"].shape[1])
        hidden_dim = int(state_dict["fc1.weight"].shape[0])
        num_classes = int(state_dict["fc2.weight"].shape[0])
        head = MLPHead(
            in_dim=in_dim,
            hidden=hidden_dim,
            num_classes=num_classes,
            dropout=0.0,
            use_ln=("ln.weight" in state_dict),
        ).to(device)
        head.load_state_dict(state_dict, strict=True)
        head.eval()
        return head

    if "weight" in state_dict and "bias" in state_dict:
        num_classes = int(state_dict["weight"].shape[0])
        in_dim = int(state_dict["weight"].shape[1])
        head = nn.Linear(in_dim, num_classes).to(device)
        head.load_state_dict(state_dict, strict=True)
        head.eval()
        return head

    raise ValueError(f"Unsupported head state format: keys={list(state_dict.keys())[:8]}")


def _prepare_single_graph_inputs(hgnn, graph: dgl.DGLHeteroGraph, device: torch.device):
    batched_graph = dgl.batch([graph])
    # 先在 CPU 侧收集 edge_index / edge_feature，避免 DGL 在 GPU 图上访问 edata 时触发懒拷贝。
    edge_index_dict = legacy_bridge._prepare_edge_indices_for_device(hgnn, batched_graph, device)
    edge_feature_dict = legacy_bridge._prepare_edge_features_for_device(hgnn, batched_graph, device)
    homo_graph = legacy_bridge._prepare_homo_graph_for_device(hgnn, batched_graph, device)
    batched_graph = batched_graph.to(device)
    return batched_graph, edge_index_dict, edge_feature_dict, homo_graph


def _build_hgmp_prompt_module(sample_graph: dgl.DGLHeteroGraph, device: torch.device) -> HeteroPrompt:
    ntypes = sample_graph.ntypes
    token_dims = [sample_graph.ndata["x"][nt].shape[1] for nt in ntypes]
    return HeteroPrompt(token_dims=token_dims, ntypes=ntypes).to(device)


def _make_args():
    parser = build_parser()
    parser.add_argument("--method", type=str, required=True, choices=["hgmp", "hgmp_prompt", "peprompt"])
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--repeat_id", type=int, default=0)
    parser.add_argument("--best_ckpt", type=str, default=None)
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=ROOT / "artifacts" / "analysis" / "micro_visualization",
    )
    parser.add_argument("--trajectory_max_samples", type=int, default=200)
    args = parser.parse_args()
    args = _normalize_dataset_defaults(args)
    ckpt_by_method = {args.method: _resolve_ckpt(args, args.method)}
    args = _sync_args_with_legacy_ckpts(args, ckpt_by_method)
    return args


def _collapse_label(label_tensor: torch.Tensor) -> int:
    label_tensor = label_tensor.detach().cpu()
    if label_tensor.ndim == 0:
        return int(label_tensor.item())
    label_id = legacy_bridge._to_class_ids(label_tensor).view(-1)
    if label_id.numel() == 0:
        raise ValueError("Label tensor is empty.")
    return int(label_id[0].item())


def _predict_confidence(logits: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError(f"Expected logits shape [1, C], got {tuple(logits.shape)}")
    if logits.size(1) == 1:
        return float(torch.sigmoid(logits).view(-1)[0].item())
    return float(torch.softmax(logits, dim=-1).max(dim=-1).values[0].item())


def _select_case(samples: list[SamplePrediction], *, correct: bool) -> SamplePrediction | None:
    subset = [sample for sample in samples if (sample.pred_label == sample.true_label) == correct]
    if not subset:
        return None
    subset.sort(key=lambda item: item.confidence, reverse=True)
    return subset[0]


def _serialize_inverse_indices(value):
    if isinstance(value, dict):
        out = {}
        for key, tensor in value.items():
            if isinstance(tensor, torch.Tensor):
                out[key] = tensor.detach().cpu().view(-1).tolist()
            else:
                out[key] = list(tensor)
        return out
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    return value


def _load_splits(legacy_args):
    train_list, val_list, test_list, targetnode = load_peprompt_offline_legacy_splits(legacy_args)
    return train_list, val_list, test_list, targetnode


def _load_models(args, legacy_args, sample_graph: dgl.DGLHeteroGraph, targetnode: str):
    encoder_ckpt = _resolve_ckpt(args, args.method)
    best_ckpt = Path(args.best_ckpt) if args.best_ckpt else _infer_best_ckpt_path(args)
    if not best_ckpt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_ckpt}")

    best_payload = _load_torch_payload(best_ckpt)

    if args.method == "hgmp":
        hgnn = legacy_bridge.build_frozen_legacy_hgnn(legacy_args, encoder_ckpt)
        head = _build_linear_or_mlp_head(best_payload["head_state"], legacy_args.device)
        adapter = LegacyModelVisualizerAdapter(
            method=args.method,
            hgnn=hgnn,
            targetnode=targetnode,
        )
        return hgnn, None, head, adapter, best_ckpt

    if args.method == "peprompt":
        hgnn = legacy_bridge.build_legacy_hgnn(
            legacy_args,
            encoder_ckpt,
            freeze=False,
            train_relation_prompt_only=False,
        )
        hgnn.load_state_dict(best_payload["hgnn_state"], strict=False)
        hgnn.eval()
        for p in hgnn.parameters():
            p.requires_grad_(False)
        head = _build_linear_or_mlp_head(best_payload["head_state"], legacy_args.device)
        adapter = LegacyModelVisualizerAdapter(
            method=args.method,
            hgnn=hgnn,
            targetnode=targetnode,
            edge_feature_name=getattr(legacy_args, "peprompt_edge_feature_name", "peprompt_edge_feat"),
        )
        return hgnn, None, head, adapter, best_ckpt

    if args.method == "hgmp_prompt":
        hgnn = legacy_bridge.build_frozen_legacy_hgnn(legacy_args, encoder_ckpt)
        prompt_module = _build_hgmp_prompt_module(sample_graph, legacy_args.device)
        prompt_module.load_state_dict(best_payload["pg_state"], strict=True)
        prompt_module.eval()
        head = _build_linear_or_mlp_head(best_payload["head_state"], legacy_args.device)
        adapter = LegacyModelVisualizerAdapter(
            method=args.method,
            hgnn=hgnn,
            targetnode=targetnode,
            prompt_module=prompt_module,
        )
        return hgnn, prompt_module, head, adapter, best_ckpt

    raise ValueError(f"Unsupported method: {args.method}")


@torch.no_grad()
def _predict_one_sample(
    args,
    legacy_args,
    hgnn,
    head: nn.Module,
    adapter: LegacyModelVisualizerAdapter,
    sample,
    targetnode: str,
    prompt_module=None,
) -> SamplePrediction:
    graph_cpu, inverse_indices, label = sample
    true_label = _collapse_label(torch.as_tensor(label))

    batched_graph, edge_index_dict, edge_feature_dict, homo_graph = _prepare_single_graph_inputs(
        hgnn,
        graph_cpu,
        legacy_args.device,
    )

    prompt_before_x_dict = None
    analysis_graph = batched_graph
    if prompt_module is not None:
        prompt_before_x_dict = {
            ntype: batched_graph.ndata["x"][ntype].detach().clone()
            for ntype in batched_graph.ntypes
        }
        analysis_graph = prompt_module(batched_graph)
        edge_index_dict = legacy_bridge._prepare_edge_indices_for_device(hgnn, analysis_graph, legacy_args.device)
        edge_feature_dict = legacy_bridge._prepare_edge_features_for_device(hgnn, analysis_graph, legacy_args.device)
        homo_graph = legacy_bridge._prepare_homo_graph_for_device(hgnn, analysis_graph, legacy_args.device)

    graph_emb = legacy_bridge.forward_graph_batch(
        hgnn,
        analysis_graph,
        targetnode,
        edge_feature_dict=edge_feature_dict,
        edge_index_dict=edge_index_dict,
        homo_graph=homo_graph,
    )
    logits = head(graph_emb)
    pred_label = int(legacy_bridge._to_class_ids(logits).view(-1)[0].item())
    confidence = _predict_confidence(logits)
    capture = adapter.capture(
        analysis_graph,
        inverse_indices,
        prompt_before_x_dict=prompt_before_x_dict,
    )

    return SamplePrediction(
        graph=graph_cpu,
        inverse_indices=inverse_indices,
        true_label=true_label,
        pred_label=pred_label,
        confidence=confidence,
        capture=capture,
    )


def _write_case_outputs(case: SamplePrediction, case_name: str, out_dir: Path):
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    visualize_subgraph_aggregation(
        subgraph=case.graph,
        target_nid=case.inverse_indices,
        attention_dict=case.capture.attention_dict,
        pe_dict=case.capture.pe_norm_dict,
        save_path=case_dir / "aggregation.png",
        title=f"{case_name} | y={case.true_label} pred={case.pred_label}",
    )

    visualize_target_feature_heatmap(
        layer_embeddings=case.capture.layer_embeddings,
        save_path=case_dir / "target_feature_heatmap.png",
        title=f"{case_name} | target feature heatmap",
    )
    visualize_target_feature_delta_bars(
        layer_embeddings=case.capture.layer_embeddings,
        save_path=case_dir / "target_feature_delta_topk.png",
        title=f"{case_name} | top feature deltas",
    )
    node_rows = visualize_node_drift_heatmap(
        node_layer_embeddings=case.capture.node_layer_embeddings,
        save_path=case_dir / "node_drift_heatmap.png",
        title=f"{case_name} | node drift metrics",
    )
    _write_dict_rows_csv(case_dir / "node_drift_metrics.csv", node_rows)

    payload = CaseRecord(
        case_name=case_name,
        true_label=case.true_label,
        pred_label=case.pred_label,
        confidence=case.confidence,
        inverse_indices=_serialize_inverse_indices(case.inverse_indices),
        prompt_summary=case.capture.prompt_summary,
        score_mode=case.capture.score_mode,
    )
    (case_dir / "summary.json").write_text(
        json.dumps(asdict(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = _make_args()
    legacy_args = _make_legacy_args(
        args,
        args.method,
        _resolve_ckpt(args, args.method),
        args.split_seed,
        args.repeat_id,
    )
    _set_global_seed(legacy_args.seed)

    _, _, test_list, targetnode = _load_splits(legacy_args)
    if len(test_list) == 0:
        raise RuntimeError("Test split is empty, cannot analyze bad cases.")

    sample_graph = test_list[0][0]
    hgnn, prompt_module, head, adapter, best_ckpt = _load_models(
        args,
        legacy_args,
        sample_graph,
        targetnode,
    )

    predictions: list[SamplePrediction] = []
    for sample in test_list:
        predictions.append(
            _predict_one_sample(
                args=args,
                legacy_args=legacy_args,
                hgnn=hgnn,
                head=head,
                adapter=adapter,
                sample=sample,
                targetnode=targetnode,
                prompt_module=prompt_module,
            )
        )

    correct_case = _select_case(predictions, correct=True)
    wrong_case = _select_case(predictions, correct=False)

    out_dir = (
        Path(args.analysis_dir)
        / args.dataset
        / args.method
        / f"{args.shot}-shot"
        / f"splitseed{args.split_seed}"
        / f"repeat{args.repeat_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = predictions[: min(len(predictions), int(args.trajectory_max_samples))]
    layer_embeddings_list = []
    labels = []
    for pred in subset:
        ordered_layers = [pred.capture.layer_embeddings[name] for name in ("Layer0", "Layer1", "Layer2") if name in pred.capture.layer_embeddings]
        if len(ordered_layers) < 2:
            continue
        layer_embeddings_list.append(ordered_layers)
        labels.append(pred.true_label)

    if layer_embeddings_list:
        visualize_feature_trajectory(
            layer_embeddings_list=layer_embeddings_list,
            labels=labels,
            save_path=out_dir / "feature_trajectory_tsne.png",
            layer_names=["Layer0", "Layer1", "Layer2"],
            title=f"{args.dataset} | {args.method} | Target Feature Trajectory",
        )

    if correct_case is not None:
        _write_case_outputs(correct_case, "true_positive", out_dir)
    if wrong_case is not None:
        _write_case_outputs(wrong_case, "misclassified", out_dir)

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "shot": int(args.shot),
        "split_seed": int(args.split_seed),
        "repeat_id": int(args.repeat_id),
        "encoder_ckpt": _resolve_ckpt(args, args.method),
        "best_ckpt": str(best_ckpt),
        "num_test_samples": int(len(predictions)),
        "num_correct": int(sum(int(x.pred_label == x.true_label) for x in predictions)),
        "num_wrong": int(sum(int(x.pred_label != x.true_label) for x in predictions)),
        "trajectory_samples": int(len(layer_embeddings_list)),
        "correct_case_available": bool(correct_case is not None),
        "wrong_case_available": bool(wrong_case is not None),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
