"""
Feature Engineering Module
Transforms candidate pairs into a structured 19-feature comparison matrix.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "blocking_score",
    "common_words_count",
    "word_jaccard",
    "is_exact_core_match",
    "is_substring_match",
    "has_state_s1",
    "has_state_cand",
    "both_have_state",
    "state_match",
    "has_house_num_s1",
    "has_house_num_cand",
    "both_have_house_num",
    "house_number_match",
    "address_common_words_count",
    "address_word_jaccard",
    "source_is_s3",
    "name_x_addr_jaccard",
    "both_exact_and_state",
    "score_per_word"
]

def compute_pair_features(s1_df, blocked_candidates, s2_df, s3_df, gt_df=None):
    """Computes the 19 comparison features for all candidate pairs."""
    gt_map = {}
    if gt_df is not None:
        for _, row in gt_df.iterrows():
            sid = row["source1_entity_id"]
            m_str = str(row.get("matched_entity_ids", ""))
            gt_map[sid] = set(m_str.split(",")) if pd.notna(m_str) and m_str.strip() else set()

    s1_lookup = s1_df.set_index("entity_id").to_dict(orient="index")
    pairs_list = []

    for sid, cands_df in blocked_candidates.items():
        if cands_df is None or len(cands_df) == 0:
            continue
        s1_row = s1_lookup.get(sid)
        if not s1_row:
            continue

        s1_name = str(s1_row.get("clean_name", "")).strip()
        s1_words = set(s1_name.split()) if s1_name else set()
        s1_addr = str(s1_row.get("clean_address", "")).strip()
        s1_addr_words = set(s1_addr.split()) if s1_addr else set()
        s1_st = s1_row.get("state")
        has_st_s1 = int(pd.notna(s1_st) and str(s1_st).strip() != "")
        s1_hn = s1_row.get("house_number")
        has_hn_s1 = int(pd.notna(s1_hn) and str(s1_hn).strip() != "")
        gt_matches = gt_map.get(sid, set())

        for _, c_row in cands_df.iterrows():
            cid = c_row["entity_id"]
            src = c_row["source"]
            c_name = str(c_row.get("clean_name", "")).strip()
            c_words = set(c_name.split()) if c_name else set()
            c_addr = str(c_row.get("clean_address", "")).strip()
            c_addr_words = set(c_addr.split()) if c_addr else set()
            c_st = c_row.get("state")
            has_st_c = int(pd.notna(c_st) and str(c_st).strip() != "")
            c_hn = c_row.get("house_number")
            has_hn_c = int(pd.notna(c_hn) and str(c_hn).strip() != "")

            common_words = len(s1_words & c_words)
            union_words = len(s1_words | c_words)
            jaccard = (common_words / union_words) if union_words > 0 else 0.0
            is_exact = int(s1_name == c_name and len(s1_name) > 0)
            is_substr = int((s1_name in c_name or c_name in s1_name) and len(s1_name) >= 3 and len(c_name) >= 3)

            both_st = int(has_st_s1 == 1 and has_st_c == 1)
            st_mat = int(has_st_s1 == 1 and has_st_c == 1 and str(s1_st).strip().lower() == str(c_st).strip().lower())

            both_hn = int(has_hn_s1 == 1 and has_hn_c == 1)
            hn_mat = int(has_hn_s1 == 1 and has_hn_c == 1 and str(s1_hn).strip().lower() == str(c_hn).strip().lower())

            addr_common = len(s1_addr_words & c_addr_words)
            addr_union = len(s1_addr_words | c_addr_words)
            addr_jaccard = (addr_common / addr_union) if addr_union > 0 else 0.0

            blocking_sc = float(c_row.get("matching_score", 0.0))
            is_true_match = int(cid in gt_matches) if gt_df is not None else None

            pair_dict = {
                "s1_id": sid,
                "candidate_id": cid,
                "source": src,
                "blocking_score": blocking_sc,
                "common_words_count": common_words,
                "word_jaccard": jaccard,
                "is_exact_core_match": is_exact,
                "is_substring_match": is_substr,
                "has_state_s1": has_st_s1,
                "has_state_cand": has_st_c,
                "both_have_state": both_st,
                "state_match": st_mat,
                "has_house_num_s1": has_hn_s1,
                "has_house_num_cand": has_hn_c,
                "both_have_house_num": both_hn,
                "house_number_match": hn_mat,
                "address_common_words_count": addr_common,
                "address_word_jaccard": addr_jaccard,
            }
            if is_true_match is not None:
                pair_dict["label"] = is_true_match
            pairs_list.append(pair_dict)

    df_pairs = pd.DataFrame(pairs_list)
    if len(df_pairs) > 0:
        df_pairs["source_is_s3"] = (df_pairs["source"] == "S3").astype(int)
        df_pairs["name_x_addr_jaccard"] = df_pairs["word_jaccard"] * df_pairs["address_word_jaccard"]
        df_pairs["both_exact_and_state"] = df_pairs["is_exact_core_match"] * df_pairs["state_match"]
        df_pairs["score_per_word"] = df_pairs["blocking_score"] / df_pairs["common_words_count"].clip(lower=1)
    return df_pairs
