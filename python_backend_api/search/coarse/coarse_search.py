import os, sys
import pandas as pd, numpy as np
import re, math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst

## Loading features and metadata beforehand
(
    cld_tfidf_df,
    cld_tfidf_np,
    edh_tfidf_df,
    edh_tfidf_np,
    hog_tfidf_df,
    hog_tfidf_np,
    text_tfidf_df,
    text_tfidf_np,
    comic_cover_img_df,
    comic_cover_img_np,
    comic_cover_txt_df,
    comic_cover_txt_np,
) = utils.load_all_coarse_features()

## Get all interpretable features for no reranked search
(
    interpretable_scaled_features_np,
    interpretable_feature_lst,
    gender_feat_np,
    supersense_feat_np,
    genre_feat_np,
    panel_ratio_feat_np,
    comic_cover_img_np,
    comic_cover_txt_np,
) = utils.load_all_interpretable_features()

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()


def get_top_n_matching_book_info(
    idx_top_n_np,
    sim_score_top_n_np,
    comic_info_dict=book_metadata_dict,
    print_n=20,
    query_book_id=1,
    feature_similarity_type="cld",
):
    sim_score_top_n_squeezed_np = np.squeeze(sim_score_top_n_np)
    list_of_records = []
    # print(comic_info_dict[query_book_id])
    query_comic_no, query_book_title, query_genre, year = comic_info_dict[query_book_id]
    # print(idx_top_n_np, sim_score_top_n_squeezed_np.shape)
    for i in range(1, print_n):

        book_idx = idx_top_n_np[i]
        sim_score_book = sim_score_top_n_squeezed_np[i]

        try:
            comic_no, book_title, genre, year = comic_info_dict[book_idx]
        except Exception as e:
            comic_no, book_title, genre, year = (-1, "not exist", "not exist", "n.a")

        list_of_records.append(
            {
                "rank": i,
                "sim_score": sim_score_book,
                "comic_no": comic_no,
                "book_title": book_title,
                "genre": genre,
                "year": year,
                "query_comic_no": query_comic_no,
                "query_book_title": query_book_title,
                "query_genre": query_genre,
                "feature_similarity_type": feature_similarity_type,
            }
        )

    df = pd.DataFrame.from_dict(list_of_records)
    return df


def comics_coarse_search(
    query_comic_book_id: int, feature_weight_dict: dict, top_n: int
):

    # remove this later
    query_book_id = query_comic_book_id  # -3451

    # get similarity for all features
    cld_cosine_similarity = utils.cosine_similarity(
        cld_tfidf_np[:, :], cld_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    edh_cosine_similarity = utils.cosine_similarity(
        edh_tfidf_np[:, :], edh_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    hog_cosine_similarity = utils.cosine_similarity(
        hog_tfidf_np[:, :], hog_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    text_cosine_similarity = utils.cosine_similarity(
        text_tfidf_np[:, :], text_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )

    comic_cover_img_cosine_similarity = utils.cosine_similarity(
        comic_cover_img_np[:, :],
        comic_cover_img_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    comic_cover_txt_cosine_similarity = utils.cosine_similarity(
        comic_cover_txt_np[:, :],
        comic_cover_txt_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    # combine similarity and weigh them
    combined_results_similarity = (
        cld_cosine_similarity * feature_weight_dict["cld"]
        + edh_cosine_similarity * feature_weight_dict["edh"]
        + hog_cosine_similarity * feature_weight_dict["hog"]
        + text_cosine_similarity * feature_weight_dict["text"]
        + comic_cover_img_cosine_similarity * feature_weight_dict["comic_img"]
        + comic_cover_txt_cosine_similarity * feature_weight_dict["comic_txt"]
    )

    # find top book indices according to combined similarity
    combined_results_indices = np.argsort(
        np.squeeze(-combined_results_similarity), axis=0
    )

    # sort indices by their combined similarity score to pick top k
    combined_sorted_result_indices = np.sort(-combined_results_similarity, axis=0)

    top_k_df = get_top_n_matching_book_info(
        idx_top_n_np=combined_results_indices,
        sim_score_top_n_np=combined_sorted_result_indices,
        comic_info_dict=book_metadata_dict,
        print_n=top_n,
        query_book_id=query_book_id,
        feature_similarity_type="coarse_combined",
    )

    return top_k_df


def comics_coarse_search_without_reranking(
    query_comic_book_id: int, feature_weight_dict: dict, top_n: int
):

    # remove this later
    query_book_id = query_comic_book_id  # -3451

    # get similarity for all coarse features
    cld_cosine_similarity = utils.cosine_similarity(
        cld_tfidf_np[:, :], cld_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    edh_cosine_similarity = utils.cosine_similarity(
        edh_tfidf_np[:, :], edh_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    hog_cosine_similarity = utils.cosine_similarity(
        hog_tfidf_np[:, :], hog_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )
    text_cosine_similarity = utils.cosine_similarity(
        text_tfidf_np[:, :], text_tfidf_np[max(query_book_id, 0) : query_book_id + 1, :]
    )

    # get similarity for all interpretable features
    gender_cosine_similarity = utils.cosine_similarity(
        gender_feat_np[:, :],
        gender_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    supersense_cosine_similarity = utils.cosine_similarity(
        supersense_feat_np[:, :],
        supersense_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    genre_cosine_similarity = utils.cosine_similarity(
        genre_feat_np[:, :],
        genre_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    panel_ratio_cosine_similarity = utils.cosine_similarity(
        panel_ratio_feat_np[:, :],
        panel_ratio_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    comic_cover_img_cosine_similarity = utils.cosine_similarity(
        comic_cover_img_np[:, :],
        comic_cover_img_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    comic_cover_txt_cosine_similarity = utils.cosine_similarity(
        comic_cover_txt_np[:, :],
        comic_cover_txt_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    # combine similarity and weigh them
    combined_results_similarity = (
        cld_cosine_similarity * feature_weight_dict["cld"]
        + edh_cosine_similarity * feature_weight_dict["edh"]
        + hog_cosine_similarity * feature_weight_dict["hog"]
        + text_cosine_similarity * feature_weight_dict["text"]
        + comic_cover_img_cosine_similarity * feature_weight_dict["comic_img"]
        + comic_cover_txt_cosine_similarity * feature_weight_dict["comic_txt"]
        + gender_cosine_similarity * feature_weight_dict["gender"]
        + supersense_cosine_similarity * feature_weight_dict["supersense"]
        + genre_cosine_similarity * feature_weight_dict["genre_comb"]
        + panel_ratio_cosine_similarity * feature_weight_dict["panel_ratio"]
    )

    # find top book indices according to combined similarity
    combined_results_indices = np.argsort(
        np.squeeze(-combined_results_similarity), axis=0
    )

    # sort indices by their combined similarity score to pick top k
    combined_sorted_result_indices = np.sort(-combined_results_similarity, axis=0)

    top_k_df = get_top_n_matching_book_info(
        idx_top_n_np=combined_results_indices,
        sim_score_top_n_np=combined_sorted_result_indices,
        comic_info_dict=book_metadata_dict,
        print_n=top_n,
        query_book_id=query_book_id,
        feature_similarity_type="coarse_combined",
    )

    return top_k_df


def comics_random_search(
    query_comic_book_id: int, feature_weight_dict: dict, top_n: int
):
    all_books_except_query_books = [
        i
        for i in range(1, comic_book_metadata_df.shape[0] - 5)
        if i != query_comic_book_id
    ]
    serp_ids_lst = np.random.choice(
        all_books_except_query_books, size=top_n, replace=False
    ).tolist()

    query_book_obj = book_metadata_dict[
        query_comic_book_id
    ]  # get querry object details
    serp_lst = [
        {
            "comic_no": book_metadata_dict[id][0],
            "book_title": book_metadata_dict[id][1],
            "genre": str(book_metadata_dict[id][2]),
            "year": book_metadata_dict[id][3]
            if not isinstance(book_metadata_dict[id][3], str)
            and not math.isnan(book_metadata_dict[id][3])
            else 1950,
            "query_book": False,
        }
        for id in serp_ids_lst
    ]

    return serp_lst


if __name__ == "__main__":
    query_book_comic_id = 50  # 1647(tin-tin), 520(aquaman), 558(asterix), 587(Avengers), 650(Batman), 1270(Justice Society)
    top_n = 21
    feature_weight_dict = {
        "cld": 0.1,
        "edh": 0.1,
        "hog": 0.1,
        "text": 1.7,
        "comic_img": 1.0,
        "comic_txt": 1.0,
    }
    # {'cld': 0.4, 'edh': 0.4, 'hog': 0.4, 'text': 0.8}
    # print(book_metadata_dict)
    print("query book info: {}".format(book_metadata_dict[query_book_comic_id]))

    top_n_results_df = comics_coarse_search(
        query_book_comic_id, feature_weight_dict=feature_weight_dict, top_n=top_n
    )
    print(top_n_results_df.head(top_n))
