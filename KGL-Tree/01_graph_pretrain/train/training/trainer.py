import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from embedding_client import EmbeddingClient
from model_defs import GraphEncoder, PathEncoder, PosProjector, QueryFusion, StepAwareQueryFusion, TaskProjector

from .candidate_utils import (
    build_candidate_bonus,
    build_candidate_set,
    build_dense_history_bonus,
    history_bonus_scale,
)
from .config import (
    BASE_DATA_DIR,
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    DYNAMIC_KEEP_RATIO,
    ENABLE_STEP1_SEMANTIC_PREFILTER,
    EPOCHS,
    HIDDEN_DIM,
    HISTORY_BONUS_LATER,
    HISTORY_BONUS_STEP1,
    HISTORY_TOPK,
    INFER_ALLOWED,
    KG_PATH,
    LABEL_SMOOTHING,
    LAST_MODEL_PATH,
    LR,
    MAX_CANDIDATES,
    MODEL_PATH,
    PATIENCE,
    REL_TYPES,
    SEED,
    SERVICE_EMB_CACHE_PATH,
    SINGLE_STEP_LOSS_WEIGHT,
    STRICT_NEIGHBOR_ONLY_AFTER_STEP1,
    TASK_EMB_CACHE_PATH,
    TASKS_FILE_PATH,
    TEMPERATURE,
    TEST_TASKS_FILE_PATH,
    VAL_RATIO,
    WEIGHT_DECAY,
    WORKFLOW_REPEAT_CAP,
    WORKFLOW_REPEAT_CAP_NO_DEP,
    DEP_REPEAT,
    USE_DEPENDENCY_FIRST_DECODING,
    USE_PATH_ENCODER,
    USE_POSITIONAL_SIGNAL,
    USE_STEP_AWARE_FUSION,
    WF_DEP_BONUS,
)
from .data_utils import (
    Sample,
    build_or_load_service_embeddings,
    build_or_load_task_emb_cache,
    build_samples,
    load_graph,
    load_task_records,
    set_seed,
    split_by_task,
)
from .history_utils import SimilarTaskRetriever, build_history_tasks


def get_task_batch(
    task_texts: List[str],
    task_cache: Dict[str, torch.Tensor],
    embed_client: EmbeddingClient,
    device: torch.device,
) -> torch.Tensor:
    batch = []
    for text in task_texts:
        if text not in task_cache:
            task_cache[text] = torch.tensor(embed_client.get_embedding(text), dtype=torch.float32, device=device)
        batch.append(task_cache[text])
    return torch.stack(batch, dim=0)


def encode_paths(
    batch_paths: List[List[str]],
    service_embs: torch.Tensor,
    idx_map: Dict[str, int],
    path_encoder: Optional[PathEncoder],
    device: torch.device,
) -> torch.Tensor:
    dim = service_embs.size(1)
    if path_encoder is None:
        return torch.zeros((len(batch_paths), dim), dtype=service_embs.dtype, device=device)

    seqs = []
    lengths = []

    for path in batch_paths:
        vecs = []
        for service_id in path:
            node = f"Service:{service_id}"
            if node in idx_map:
                vecs.append(service_embs[idx_map[node]])

        if not vecs:
            vecs = [torch.zeros(dim, device=device)]

        seqs.append(torch.stack(vecs))
        lengths.append(len(vecs))

    padded = pad_sequence(seqs, batch_first=True)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    return path_encoder(padded, lengths_t)


def get_prev_batch(
    prev_nodes: List[Optional[str]],
    service_embs: torch.Tensor,
    idx_map: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    dim = service_embs.size(1)
    batch = []
    for node in prev_nodes:
        if node is None or node not in idx_map:
            batch.append(torch.zeros(dim, device=device))
        else:
            batch.append(service_embs[idx_map[node]])
    return torch.stack(batch, dim=0)


def get_pos_batch(samples: List[Sample], device: torch.device) -> torch.Tensor:
    features = []
    for sample in samples:
        total = max(1, int(sample.seq_len))
        step = max(1, int(sample.step_idx))
        features.append([float(step) / total, float(total - step) / total])
    return torch.tensor(features, dtype=torch.float32, device=device)


def fuse_query(
    query_fusion: nn.Module,
    task_emb: torch.Tensor,
    path_emb: torch.Tensor,
    prev_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    step_idx_t: torch.Tensor,
    seq_len_t: torch.Tensor,
) -> torch.Tensor:
    if USE_STEP_AWARE_FUSION:
        return query_fusion(task_emb, path_emb, prev_emb, pos_emb, step_idx_t, seq_len_t)
    if USE_POSITIONAL_SIGNAL:
        return query_fusion(task_emb, path_emb, prev_emb, pos_emb)
    return query_fusion(task_emb, path_emb, prev_emb)


def build_checkpoint_payload(
    encoder,
    path_encoder,
    task_proj,
    pos_proj,
    query_fusion,
    idx_map,
    service_nodes,
    num_relations,
    relation_vocab,
    train_task_ids,
    metrics,
):
    return {
        "encoder": encoder.state_dict(),
        "path_encoder": path_encoder.state_dict() if path_encoder is not None else None,
        "task_proj": task_proj.state_dict(),
        "pos_proj": pos_proj.state_dict() if pos_proj is not None else None,
        "query_fusion": query_fusion.state_dict(),
        "idx_map": idx_map,
        "service_nodes": service_nodes,
        "num_relations": num_relations,
        "relation_vocab": relation_vocab,
        "enabled_rel_types": list(REL_TYPES),
        "hidden_dim": HIDDEN_DIM,
        "infer_allowed_relations": list(INFER_ALLOWED),
        "metrics_val_best": metrics,
        "use_positional_signal": USE_POSITIONAL_SIGNAL,
        "use_step_aware_fusion": USE_STEP_AWARE_FUSION,
        "use_path_encoder": USE_PATH_ENCODER,
        "use_dependency_first_decoding": USE_DEPENDENCY_FIRST_DECODING,
        "dep_priority": {
            "DEP_REPEAT": DEP_REPEAT,
            "WF_DEP_BONUS": WF_DEP_BONUS,
            "WORKFLOW_REPEAT_CAP": WORKFLOW_REPEAT_CAP,
            "WORKFLOW_REPEAT_CAP_NO_DEP": WORKFLOW_REPEAT_CAP_NO_DEP,
        },
        "dep_single_direction": True,
        "start_from_all_services": True,
        "query_fusion_type": "step_aware" if USE_STEP_AWARE_FUSION else "query_fusion",
        "query_fusion_n_inputs": 4 if USE_POSITIONAL_SIGNAL else 3,
        "single_step_loss_weight": SINGLE_STEP_LOSS_WEIGHT,
        "step1_semantic_prefilter": ENABLE_STEP1_SEMANTIC_PREFILTER,
        "step1_semantic_topk_dynamic_ratio": DYNAMIC_KEEP_RATIO,
        "history_topk": HISTORY_TOPK,
        "history_bonus_step1": HISTORY_BONUS_STEP1,
        "history_bonus_later": HISTORY_BONUS_LATER,
        "history_train_task_ids": sorted(train_task_ids),
        "history_tasks_file": TASKS_FILE_PATH,
        "strict_neighbor_only_after_step1": STRICT_NEIGHBOR_ONLY_AFTER_STEP1,
    }


@torch.no_grad()
def evaluate(
    samples,
    encoder,
    path_encoder,
    task_proj,
    pos_proj,
    query_fusion,
    service_text_embs,
    edge_index,
    edge_type,
    service_nodes,
    idx_map,
    task_cache,
    dep_neighbors,
    fallback_neighbors,
    start_nodes,
    history_retriever,
    embed_client,
    desc: str,
):
    encoder.eval()
    if path_encoder is not None:
        path_encoder.eval()
    task_proj.eval()
    if pos_proj is not None:
        pos_proj.eval()
    query_fusion.eval()

    service_embs = encoder(service_text_embs, edge_index, edge_type)

    hit1_global = 0
    hit3_global = 0
    mrr_global = 0.0
    hit1_cons = 0
    hit3_cons = 0
    mrr_cons = 0.0
    valid_cons = 0

    for sample in tqdm(samples, desc=desc, leave=False):
        task_emb = task_proj(get_task_batch([sample.task_text], task_cache, embed_client, DEVICE))
        path_emb = encode_paths([sample.path_service_ids], service_embs, idx_map, path_encoder, DEVICE)
        prev_emb = get_prev_batch([sample.prev_service_node], service_embs, idx_map, DEVICE)
        if pos_proj is not None:
            pos_emb = pos_proj(get_pos_batch([sample], DEVICE))
        else:
            pos_emb = torch.zeros_like(task_emb)

        step_idx_t = torch.tensor([sample.step_idx], dtype=torch.long, device=DEVICE)
        seq_len_t = torch.tensor([sample.seq_len], dtype=torch.long, device=DEVICE)
        query = fuse_query(query_fusion, task_emb, path_emb, prev_emb, pos_emb, step_idx_t, seq_len_t)

        history_prior = history_retriever.get_service_prior(sample.task_id, sample.task_text, task_cache)
        bonus_scale = history_bonus_scale(sample.step_idx, HISTORY_BONUS_STEP1, HISTORY_BONUS_LATER)
        dense_bonus = build_dense_history_bonus(service_nodes, history_prior, bonus_scale, DEVICE)

        target = idx_map[sample.target_service_node]
        logits = torch.matmul(query, service_embs.t()).squeeze(0) + dense_bonus
        rank = torch.argsort(logits, descending=True)
        rpos = (rank == target).nonzero(as_tuple=False).item() + 1
        hit1_global += 1 if rpos <= 1 else 0
        hit3_global += 1 if rpos <= 3 else 0
        mrr_global += 1.0 / rpos

        cand_ids, label, valid = build_candidate_set(
            sample=sample,
            idx_map=idx_map,
            dep_neighbors=dep_neighbors,
            fallback_neighbors=fallback_neighbors,
            start_nodes=start_nodes,
            task_emb=task_emb,
            query_vec=query,
            service_embs=service_embs,
            service_nodes=service_nodes,
            history_prior=history_prior,
            max_candidates=MAX_CANDIDATES,
            enable_step1_prefilter=ENABLE_STEP1_SEMANTIC_PREFILTER,
            step1_keep_ratio=DYNAMIC_KEEP_RATIO,
            history_bonus_step1=HISTORY_BONUS_STEP1,
            history_bonus_later=HISTORY_BONUS_LATER,
            use_dependency_first=USE_DEPENDENCY_FIRST_DECODING,
        )

        if not valid:
            continue

        valid_cons += 1
        cand_t = torch.tensor(cand_ids, dtype=torch.long, device=DEVICE)
        bonus = build_candidate_bonus(cand_ids, service_nodes, history_prior, bonus_scale, DEVICE)
        clogits = torch.matmul(query, service_embs[cand_t].t()).squeeze(0) + bonus
        crank = torch.argsort(clogits, descending=True)
        crpos = (crank == label).nonzero(as_tuple=False).item() + 1
        hit1_cons += 1 if crpos <= 1 else 0
        hit3_cons += 1 if crpos <= 3 else 0
        mrr_cons += 1.0 / crpos

    total = max(1, len(samples))
    return {
        "hit1_global": hit1_global / total,
        "hit3_global": hit3_global / total,
        "mrr_global": mrr_global / total,
        "hit1_cons": hit1_cons / total,
        "hit3_cons": hit3_cons / total,
        "mrr_cons": mrr_cons / total,
        "constrained_valid_rate": valid_cons / total,
    }


def train(run_final_test: bool = True):
    set_seed(SEED)
    embed_client = EmbeddingClient()

    print(f"Device: {DEVICE}")
    print(f"Base data dir: {BASE_DATA_DIR}")
    print(
        f"StepAwareFusion={USE_STEP_AWARE_FUSION} | PathEncoder={USE_PATH_ENCODER} "
        f"| PositionalSignal={USE_POSITIONAL_SIGNAL} | DependencyFirst={USE_DEPENDENCY_FIRST_DECODING} "
        f"| single_step_loss_weight={SINGLE_STEP_LOSS_WEIGHT}"
    )
    print(f"Step1 semantic prefilter(dynamic half): {ENABLE_STEP1_SEMANTIC_PREFILTER}")
    print(
        f"History USES prior: topk={HISTORY_TOPK}, "
        f"bonus_step1={HISTORY_BONUS_STEP1}, bonus_later={HISTORY_BONUS_LATER}"
    )

    (
        kg,
        service_nodes,
        idx_map,
        edge_index,
        edge_type,
        num_relations,
        relation_vocab,
        dep_neighbors,
        fallback_neighbors,
        start_nodes,
        uses_by_task,
    ) = load_graph(
        kg_path=KG_PATH,
        rel_types=REL_TYPES,
        dep_repeat=DEP_REPEAT,
        wf_dep_bonus=WF_DEP_BONUS,
        workflow_repeat_cap=WORKFLOW_REPEAT_CAP,
        workflow_repeat_cap_no_dep=WORKFLOW_REPEAT_CAP_NO_DEP,
        device=DEVICE,
    )

    service_text_embs = build_or_load_service_embeddings(
        service_nodes=service_nodes,
        kg=kg,
        cache_path=SERVICE_EMB_CACHE_PATH,
        embed_client=embed_client,
        device=DEVICE,
    )

    trainval_records = load_task_records(TASKS_FILE_PATH)
    trainval_samples = build_samples(idx_map, TASKS_FILE_PATH)
    test_records = {}
    test_samples = []
    if run_final_test:
        test_records = load_task_records(TEST_TASKS_FILE_PATH)
        test_samples = build_samples(idx_map, TEST_TASKS_FILE_PATH)
    train_samples, val_samples, train_task_ids, val_task_ids = split_by_task(trainval_samples, VAL_RATIO, SEED)

    all_texts = {record.task_text for record in trainval_records.values()}
    if run_final_test:
        all_texts |= {record.task_text for record in test_records.values()}
    all_texts = sorted(all_texts)
    task_cache = build_or_load_task_emb_cache(all_texts, TASK_EMB_CACHE_PATH, embed_client, DEVICE)

    history_tasks = build_history_tasks(trainval_records, train_task_ids, uses_by_task, idx_map)
    history_retriever = SimilarTaskRetriever(history_tasks, task_cache, HISTORY_TOPK, DEVICE)

    print(
        f"Samples | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
        f"| history_tasks(train-only)={len(history_tasks)} | val_task_count={len(val_task_ids)}"
    )

    in_dim = service_text_embs.size(1)
    encoder = GraphEncoder(in_dim, HIDDEN_DIM, num_relations, DROPOUT).to(DEVICE)
    path_encoder = PathEncoder(HIDDEN_DIM, dropout=DROPOUT).to(DEVICE) if USE_PATH_ENCODER else None
    task_proj = TaskProjector(in_dim, HIDDEN_DIM).to(DEVICE)
    pos_proj = PosProjector(HIDDEN_DIM).to(DEVICE) if USE_POSITIONAL_SIGNAL else None
    if USE_STEP_AWARE_FUSION:
        query_fusion = StepAwareQueryFusion(HIDDEN_DIM).to(DEVICE)
    else:
        query_inputs = 4 if USE_POSITIONAL_SIGNAL else 3
        query_fusion = QueryFusion(HIDDEN_DIM, n_inputs=query_inputs).to(DEVICE)

    params = list(encoder.parameters()) + list(task_proj.parameters()) + list(query_fusion.parameters())
    if path_encoder is not None:
        params += list(path_encoder.parameters())
    if pos_proj is not None:
        params += list(pos_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    patience = 0
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        if path_encoder is not None:
            path_encoder.train()
        task_proj.train()
        if pos_proj is not None:
            pos_proj.train()
        query_fusion.train()

        random.shuffle(train_samples)
        total_loss = 0.0
        trained_steps = 0
        skipped_steps = 0

        progress = tqdm(range(0, len(train_samples), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}")
        for start in progress:
            batch = train_samples[start:start + BATCH_SIZE]
            if not batch:
                continue

            service_embs = encoder(service_text_embs, edge_index, edge_type)
            tasks = [sample.task_text for sample in batch]
            paths = [sample.path_service_ids for sample in batch]
            prevs = [sample.prev_service_node for sample in batch]

            task_emb = task_proj(get_task_batch(tasks, task_cache, embed_client, DEVICE))
            path_emb = encode_paths(paths, service_embs, idx_map, path_encoder, DEVICE)
            prev_emb = get_prev_batch(prevs, service_embs, idx_map, DEVICE)
            if pos_proj is not None:
                pos_emb = pos_proj(get_pos_batch(batch, DEVICE))
            else:
                pos_emb = torch.zeros_like(task_emb)
            step_idx_t = torch.tensor([sample.step_idx for sample in batch], dtype=torch.long, device=DEVICE)
            seq_len_t = torch.tensor([sample.seq_len for sample in batch], dtype=torch.long, device=DEVICE)

            query = fuse_query(query_fusion, task_emb, path_emb, prev_emb, pos_emb, step_idx_t, seq_len_t)

            loss_terms = []
            weights = []
            for idx, sample in enumerate(batch):
                history_prior = history_retriever.get_service_prior(sample.task_id, sample.task_text, task_cache)
                cand_ids, label, valid = build_candidate_set(
                    sample=sample,
                    idx_map=idx_map,
                    dep_neighbors=dep_neighbors,
                    fallback_neighbors=fallback_neighbors,
                    start_nodes=start_nodes,
                    task_emb=task_emb[idx:idx + 1],
                    query_vec=query[idx:idx + 1],
                    service_embs=service_embs,
                    service_nodes=service_nodes,
                    history_prior=history_prior,
                    max_candidates=MAX_CANDIDATES,
                    enable_step1_prefilter=ENABLE_STEP1_SEMANTIC_PREFILTER,
                    step1_keep_ratio=DYNAMIC_KEEP_RATIO,
                    history_bonus_step1=HISTORY_BONUS_STEP1,
                    history_bonus_later=HISTORY_BONUS_LATER,
                    use_dependency_first=USE_DEPENDENCY_FIRST_DECODING,
                )

                if not valid:
                    skipped_steps += 1
                    continue

                cand_t = torch.tensor(cand_ids, dtype=torch.long, device=DEVICE)
                raw_scores = torch.matmul(query[idx:idx + 1], service_embs[cand_t].t())
                bonus_scale = history_bonus_scale(sample.step_idx, HISTORY_BONUS_STEP1, HISTORY_BONUS_LATER)
                bonus = build_candidate_bonus(cand_ids, service_nodes, history_prior, bonus_scale, DEVICE).unsqueeze(0)
                logits = (raw_scores + bonus) / TEMPERATURE

                target = torch.tensor([label], dtype=torch.long, device=DEVICE)
                loss_item = F.cross_entropy(logits, target, label_smoothing=LABEL_SMOOTHING)
                loss_terms.append(loss_item)
                weights.append(SINGLE_STEP_LOSS_WEIGHT if int(sample.seq_len) == 1 else 1.0)
                trained_steps += 1

            if not loss_terms:
                progress.set_postfix(loss="skip", skipped=skipped_steps)
                continue

            loss_t = torch.stack(loss_terms)
            weight_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            loss = (loss_t * weight_t).sum() / weight_t.sum().clamp(min=1e-8)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(loss_terms)
            progress.set_postfix(loss=f"{loss.item():.4f}", used=trained_steps, skipped=skipped_steps)

        metrics = evaluate(
            samples=val_samples,
            encoder=encoder,
            path_encoder=path_encoder,
            task_proj=task_proj,
            pos_proj=pos_proj,
            query_fusion=query_fusion,
            service_text_embs=service_text_embs,
            edge_index=edge_index,
            edge_type=edge_type,
            service_nodes=service_nodes,
            idx_map=idx_map,
            task_cache=task_cache,
            dep_neighbors=dep_neighbors,
            fallback_neighbors=fallback_neighbors,
            start_nodes=start_nodes,
            history_retriever=history_retriever,
            embed_client=embed_client,
            desc="Validation",
        )

        avg_loss = total_loss / max(1, trained_steps)
        score = metrics["hit1_cons"]

        print(f"\nEpoch {epoch} | loss={avg_loss:.4f} | trained_steps={trained_steps} | skipped_steps={skipped_steps}")
        print(
            f"Val global: H@1={metrics['hit1_global']:.4f}, H@3={metrics['hit3_global']:.4f}, MRR={metrics['mrr_global']:.4f}\n"
            f"Val constr: H@1={metrics['hit1_cons']:.4f}, H@3={metrics['hit3_cons']:.4f}, "
            f"MRR={metrics['mrr_cons']:.4f}, valid_rate={metrics['constrained_valid_rate']:.4f}"
        )

        if score >= best_score:
            best_score = score
            patience = 0
            best_metrics = metrics
            torch.save(
                build_checkpoint_payload(
                    encoder,
                    path_encoder,
                    task_proj,
                    pos_proj,
                    query_fusion,
                    idx_map,
                    service_nodes,
                    num_relations,
                    relation_vocab,
                    train_task_ids,
                    metrics,
                ),
                MODEL_PATH,
            )
            print(f"Saved BEST model, constrained H@1={best_score:.4f}")
        else:
            patience += 1
            print(f"No improve. patience={patience}/{PATIENCE}")

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    torch.save(
        build_checkpoint_payload(
            encoder,
            path_encoder,
            task_proj,
            pos_proj,
            query_fusion,
            idx_map,
            service_nodes,
            num_relations,
            relation_vocab,
            train_task_ids,
            best_metrics if best_metrics is not None else {},
        ),
        LAST_MODEL_PATH,
    )
    print(f"Training done. Best constrained H@1={best_score:.4f}")

    test_metrics = None
    if run_final_test:
        print("\nTesting on TEST set with BEST checkpoint...")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        if path_encoder is not None and ckpt.get("path_encoder") is not None:
            path_encoder.load_state_dict(ckpt["path_encoder"])
        task_proj.load_state_dict(ckpt["task_proj"])
        if pos_proj is not None and ckpt.get("pos_proj") is not None:
            pos_proj.load_state_dict(ckpt["pos_proj"])
        query_fusion.load_state_dict(ckpt["query_fusion"])

        test_metrics = evaluate(
            samples=test_samples,
            encoder=encoder,
            path_encoder=path_encoder,
            task_proj=task_proj,
            pos_proj=pos_proj,
            query_fusion=query_fusion,
            service_text_embs=service_text_embs,
            edge_index=edge_index,
            edge_type=edge_type,
            service_nodes=service_nodes,
            idx_map=idx_map,
            task_cache=task_cache,
            dep_neighbors=dep_neighbors,
            fallback_neighbors=fallback_neighbors,
            start_nodes=start_nodes,
            history_retriever=history_retriever,
            embed_client=embed_client,
            desc="Test",
        )

        print(
            f"Test global: H@1={test_metrics['hit1_global']:.4f}, H@3={test_metrics['hit3_global']:.4f}, "
            f"MRR={test_metrics['mrr_global']:.4f}\n"
            f"Test constr: H@1={test_metrics['hit1_cons']:.4f}, H@3={test_metrics['hit3_cons']:.4f}, "
            f"MRR={test_metrics['mrr_cons']:.4f}, valid_rate={test_metrics['constrained_valid_rate']:.4f}"
        )

    return {
        "model_path": MODEL_PATH,
        "last_model_path": LAST_MODEL_PATH,
        "best_score": best_score,
        "best_val_metrics": best_metrics if best_metrics is not None else {},
        "test_metrics": test_metrics if test_metrics is not None else {},
        "train_task_count": len(train_task_ids),
        "val_task_count": len(val_task_ids),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "test_sample_count": len(test_samples),
    }
