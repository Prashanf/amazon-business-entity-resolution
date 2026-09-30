"""
End-to-End Inference Pipeline
Processes test sets, runs GPU blocking, extracts pairwise features,
evaluates GBDT model, and outputs official competition submission TSVs.
"""

import os
import gc
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm.auto import tqdm

from .preprocessing import preprocess_dataset
from .gpu_blocking import block_candidates_gpu
from .feature_engineering import compute_pair_features, FEATURE_COLS

def get_cached_or_preprocess(file_path, name, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"✓ Loading preprocessed {name} from disk cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(df):,} records from cache in seconds!")
        return df

    print(f"Loading raw TSV for {name}: {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    print(f"Preprocessing {name} ({len(df):,} records)...")
    preprocess_dataset(df, name)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"✓ Saved preprocessed cache to: {cache_path}")
    return df

def run_inference(
    test_dir="dataset/test",
    output_dir="output",
    model_path="models/lgbm_entity_resolver_v1.txt",
    cache_dir="data/processed",
    threshold=0.75,
    outer_block_size=50000,
    s1_batch_size=500,
    top_k=15
):
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Loading test datasets from: {test_dir} (Cache dir: {cache_dir})")
    s1_cache = os.path.join(cache_dir, "test_source1_clean.parquet") if cache_dir else None
    s2_cache = os.path.join(cache_dir, "test_source2_clean.parquet") if cache_dir else None
    s3_cache = os.path.join(cache_dir, "test_source3_clean.parquet") if cache_dir else None

    s1_full = get_cached_or_preprocess(os.path.join(test_dir, "test_source1.tsv"), "Test S1 (Reference)", s1_cache)
    s2_full = get_cached_or_preprocess(os.path.join(test_dir, "test_source2.tsv"), "Test S2 (Candidates)", s2_cache)
    s3_full = get_cached_or_preprocess(os.path.join(test_dir, "test_source3.tsv"), "Test S3 (Candidates)", s3_cache)

    print(f"✓ Datasets Ready: S1={len(s1_full):,} | S2={len(s2_full):,} | S3={len(s3_full):,}")

    print(f"Loading model booster: {model_path}")
    booster = lgb.Booster(model_file=model_path)

    # Initialize output TSV headers
    with open(matching_path, "w", encoding="utf-8") as f_m, open(candidate_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    n_s1_total = len(s1_full)
    outer_blocks = list(range(0, n_s1_total, outer_block_size))
    print(f"\nStreaming inference in {len(outer_blocks)} outer block(s) of size {outer_block_size:,}...")

    total_matches = 0
    total_singletons = 0

    for b_idx, start_idx in enumerate(outer_blocks, 1):
        end_idx = min(start_idx + outer_block_size, n_s1_total)
        s1_chunk = s1_full.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"\n[Block {b_idx}/{len(outer_blocks)}] Processing S1 records {start_idx:,} to {end_idx:,} ({len(s1_chunk):,})...")

        blocked_cands = block_candidates_gpu(
            s1_chunk, s2_full, s3_full,
            s1_batch_size=s1_batch_size,
            cand_chunk_size=500000,
            top_k=top_k
        )

        pairs_df = compute_pair_features(s1_chunk, blocked_cands, s2_full, s3_full, gt_df=None)
        
        matched_dict = {}
        candidate_dict = {}

        if len(pairs_df) > 0:
            X_test = pairs_df[FEATURE_COLS]
            probs = booster.predict(X_test)
            pairs_df["match_prob"] = probs
            pairs_df["is_match"] = (probs >= threshold).astype(int)

            matched_sub = pairs_df[pairs_df["is_match"] == 1]
            for sid, grp in matched_sub.groupby("s1_id"):
                matched_dict[sid] = grp.sort_values(by="match_prob", ascending=False)["candidate_id"].drop_duplicates().tolist()

            for sid, grp in pairs_df.groupby("s1_id"):
                candidate_dict[sid] = grp["candidate_id"].drop_duplicates().tolist()

        # Append rows directly to TSVs
        with open(matching_path, "a", encoding="utf-8") as f_m, open(candidate_path, "a", encoding="utf-8") as f_c:
            for sid in s1_chunk["entity_id"]:
                m_list = matched_dict.get(sid, [])
                c_list = candidate_dict.get(sid, [])
                f_m.write(f"{sid}\t{','.join(m_list)}\n")
                f_c.write(f"{sid}\t{','.join(c_list)}\n")
                if m_list:
                    total_matches += len(m_list)
                else:
                    total_singletons += 1

        del blocked_cands, pairs_df, matched_dict, candidate_dict
        gc.collect()

    print(f"\n✓ Full inference completed successfully!")
    print(f"Total Matches Predicted : {total_matches:,}")
    print(f"Total Singletons        : {total_singletons:,} ({total_singletons / n_s1_total * 100:.1f}%)")
    print(f"Output saved to: {matching_path} and {candidate_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source Business ER Inference Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test directory containing TSVs")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--cache-dir", default="data/processed", help="Path to preprocessed parquet disk cache")
    parser.add_argument("--model-path", default="models/lgbm_entity_resolver_v1.txt", help="Path to trained LightGBM booster")
    parser.add_argument("--threshold", type=float, default=0.75, help="Decision cutoff probability")
    parser.add_argument("--batch-size", type=int, default=500, help="S1 GPU batch size")
    args = parser.parse_args()

    run_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        s1_batch_size=args.batch_size
    )
