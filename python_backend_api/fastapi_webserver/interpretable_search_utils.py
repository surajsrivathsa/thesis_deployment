import os, sys, math
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils
import search.interpretable.interpretable_search as interpretable_search

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)


def adaptive_rerank_coarse_search_results(
    normalized_feature_importance_dict: dict,
    coarse_search_results_lst: list,
    query_comic_book_id: int,
    top_k=20,
    historical_book_ids_lst=[],
):

    (
        interpretable_search_top_k_df,
        historical_book_ids_lst,
    ) = interpretable_search.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        coarse_search_results_lst=coarse_search_results_lst,
        query_comic_book_id=query_comic_book_id,
        top_k=top_k,
        historical_book_ids_lst=historical_book_ids_lst,
    )

    interpretable_filtered_book_df = interpretable_search_top_k_df[
        ["comic_no", "book_title", "genre", "year"]
    ]

    query_book_obj = book_metadata_dict[query_comic_book_id]
    print()
    print(" =========== ============= ========== ")
    print()
    print(
        "query_comic_book_id: {} | query book object: {}".format(
            query_comic_book_id, query_book_obj
        )
    )
    print()
    print(" =========== ============= ========== ")
    print()

    interpretable_filtered_book_df.fillna("", inplace=True)

    interpretable_filtered_book_lst = interpretable_filtered_book_df.to_dict(
        "records"
    ).copy()
    interpretable_filtered_book_lst.insert(
        7,
        {
            "comic_no": query_book_obj[0],
            "book_title": query_book_obj[1],
            "genre": str(query_book_obj[2]),
            "year": query_book_obj[3]
            if not isinstance(query_book_obj[3], str)
            and not math.isnan(query_book_obj[3])
            else 1950,
            "query_book": True,
        },
    )

    interpretable_filtered_book_new_lst = []
    print(interpretable_filtered_book_lst[:2])

    for idx, d in enumerate(interpretable_filtered_book_lst):
        d["id"] = idx
        if "query_book" not in d:
            d["query_book"] = False
        d["thumbsUp"] = 0
        d["thumbsDown"] = 0

        interpretable_filtered_book_new_lst.append(d)

    # new_interpretable_filtered_book_new_lst = (
    #     interpretable_filtered_book_new_lst[:7].append(
    #         {
    #             "comic_no": query_book_obj[0],
    #             "book_title": query_book_obj[1],
    #             "genre": query_book_obj[2],
    #             "year": query_book_obj[3],
    #         }
    #     )
    #     + interpretable_filtered_book_new_lst[7:14]
    # )

    return (
        interpretable_filtered_book_new_lst,
        interpretable_filtered_book_df,
        historical_book_ids_lst,
    )

