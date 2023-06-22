import os, sys
import pandas as pd, numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils
from search.coarse import coarse_search

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)


def perform_coarse_search(
    b_id: int,
    feature_weight_dict={
        "cld": 0.1,
        "edh": 0.1,
        "hog": 0.1,
        "text": 1.0,
        "comic_img": 1.0,
        "comic_txt": 1.0,
    },
    top_n=200,
):

    # query_book_comic_id = b_id # 1262 # 1647(tin-tin), 520(aquaman), 558(asterix), 587(Avengers), 650(Batman), 1270(Justice Society)

    top_n_results_df = coarse_search.comics_coarse_search(
        query_comic_book_id=b_id, feature_weight_dict=feature_weight_dict, top_n=top_n
    )
    coarse_filtered_book_df = top_n_results_df[
        ["comic_no", "book_title", "genre", "year"]
    ]

    coarse_filtered_book_df.fillna("", inplace=True)

    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        coarse_filtered_book_new_lst.append(d)

    print("Query Book : {} ".format(b_id))
    # return (coarse_filtered_book_new_lst, coarse_filtered_book_df)
    return (coarse_filtered_book_new_lst, coarse_filtered_book_df)


def perform_coarse_search_without_reranking(
    b_id: int, feature_weight_dict: dict, top_n: int
):
    top_n_results_df = coarse_search.comics_coarse_search_without_reranking(
        query_comic_book_id=b_id, feature_weight_dict=feature_weight_dict, top_n=top_n
    )
    coarse_filtered_book_df = top_n_results_df[
        ["comic_no", "book_title", "genre", "year"]
    ]

    query_book_obj = book_metadata_dict[b_id]  # get querry object details

    coarse_filtered_book_df.fillna("", inplace=True)

    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        coarse_filtered_book_new_lst.append(d)

    print("Query Book : {} ".format(b_id))

    coarse_filtered_book_lst.insert(
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

    coarse_filtered_book_new_lst = []
    print(coarse_filtered_book_lst[:2])

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        if "query_book" not in d:
            d["query_book"] = False
        d["thumbsUp"] = 0
        d["thumbsDown"] = 0

        coarse_filtered_book_new_lst.append(d)
    # return (coarse_filtered_book_new_lst, coarse_filtered_book_df)
    return (coarse_filtered_book_new_lst, coarse_filtered_book_df)


def perform_random_search(b_id: int, feature_weight_dict: dict, top_n: int):
    top_n_results_dict = coarse_search.comics_random_search(
        query_comic_book_id=b_id, feature_weight_dict=feature_weight_dict, top_n=top_n
    )

    query_book_obj = book_metadata_dict[b_id]  # get query object details
    coarse_filtered_book_lst = []
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(top_n_results_dict):
        coarse_filtered_book_lst.append(d)

    coarse_filtered_book_lst.insert(
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

    print(coarse_filtered_book_lst[:2], len(coarse_filtered_book_lst))

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        if "query_book" not in d:
            d["query_book"] = False
        d["thumbsUp"] = 0
        d["thumbsDown"] = 0

        coarse_filtered_book_new_lst.append(d)

    coarse_filtered_book_df = pd.DataFrame.from_records(coarse_filtered_book_new_lst)
    return (coarse_filtered_book_new_lst, coarse_filtered_book_df)

