import os, sys
from pathlib import Path
import pandas as pd, numpy as np
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst

# import triplet_loss

# from supervised_feature_importance import (
#     supervised_feature_importance_using_different_models,
# )
# from unsupervised_feature_importance import (
#     unsupervised_feature_importance_from_laplacian,
# )

## Loading features and metadata beforehand
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
print(interpretable_scaled_features_np.shape)
# print(interpretable_scaled_features_np[:5, :])


def get_top_n_matching_book_info(
    idx_top_n_np,
    sim_score_top_n_np,
    comic_info_dict=book_metadata_dict,
    print_n=20,
    query_book_id=1,
    feature_similarity_type="interpretable",
):
    sim_score_top_n_squeezed_np = np.squeeze(sim_score_top_n_np)
    list_of_records = []
    query_comic_no, query_book_title, query_genre, year = comic_info_dict[query_book_id]
    # print(idx_top_n_np, sim_score_top_n_np)
    for i in range(1, print_n):

        book_idx = idx_top_n_np[i]
        sim_score_book = sim_score_top_n_squeezed_np[i]

        try:
            comic_no, book_title, genre, year = comic_info_dict[book_idx]
        except Exception as e:
            comic_no, book_title, genre, year = (-1, "not exist", "not exist")

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


def adaptive_rerank_coarse_search_results(
    normalized_feature_importance_dict: dict,
    coarse_search_results_lst: list,
    query_comic_book_id: int,
    top_k=20,
    historical_book_ids_lst=[],
):

    # get corase results idx
    coarse_filtered_book_idx_lst = [d["comic_no"] for d in coarse_search_results_lst]

    # book history ids list
    historical_book_ids_dict = dict(Counter(historical_book_ids_lst))
    print(" historical_book_ids_dict: {} ".format(historical_book_ids_dict))

    # initialize similarity reduction to reduce similarity of already historical books
    reduce_similarity_val_lst = [0.0 for x in coarse_filtered_book_idx_lst]
    for idx, coarse_result_book_id in enumerate(coarse_filtered_book_idx_lst):
        if coarse_result_book_id in historical_book_ids_dict:
            reduce_similarity_val_lst[idx] = (
                historical_book_ids_dict[coarse_result_book_id] * -0.3
            )
            # if book has been shown thrice then it will be heavily discriminated against if it has been shown only once
        else:
            pass

    reduce_similarity_val_np = np.expand_dims(
        np.array(reduce_similarity_val_lst), axis=1
    )

    # remove this later
    query_book_id = query_comic_book_id  # -3451

    # get similarity for all features
    gender_cosine_similarity = utils.cosine_similarity(
        gender_feat_np[coarse_filtered_book_idx_lst, :],
        gender_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    supersense_cosine_similarity = utils.cosine_similarity(
        supersense_feat_np[coarse_filtered_book_idx_lst, :],
        supersense_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    genre_cosine_similarity = utils.cosine_similarity(
        genre_feat_np[coarse_filtered_book_idx_lst, :],
        genre_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    panel_ratio_cosine_similarity = utils.cosine_similarity(
        panel_ratio_feat_np[coarse_filtered_book_idx_lst, :],
        panel_ratio_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    comic_cover_img_cosine_similarity = utils.cosine_similarity(
        comic_cover_img_np[coarse_filtered_book_idx_lst, :],
        comic_cover_img_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    comic_cover_txt_cosine_similarity = utils.cosine_similarity(
        comic_cover_txt_np[coarse_filtered_book_idx_lst, :],
        comic_cover_txt_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    # print(gender_cosine_similarity.shape,supersense_cosine_similarity.shape, genre_cosine_similarity.shape,  panel_ratio_cosine_similarity.shape)
    # combine similarity and weigh them
    combined_results_similarity = (
        gender_cosine_similarity * normalized_feature_importance_dict["gender"]
        + supersense_cosine_similarity
        * normalized_feature_importance_dict["supersense"]
        + genre_cosine_similarity * normalized_feature_importance_dict["genre_comb"]
        + panel_ratio_cosine_similarity
        * normalized_feature_importance_dict["panel_ratio"]
        + comic_cover_img_cosine_similarity
        * normalized_feature_importance_dict["comic_cover_img"]
        + comic_cover_txt_cosine_similarity
        * normalized_feature_importance_dict["comic_cover_txt"]
    )

    combined_results_similarity = np.add(
        reduce_similarity_val_np, combined_results_similarity
    )

    # find top book indices according to combined similarity
    combined_results_indices_idx = np.argsort(
        np.squeeze(-combined_results_similarity), axis=0
    )

    combined_results_indices = np.asarray(
        [
            coarse_filtered_book_idx_lst[ranked_idx]
            for ranked_idx in list(combined_results_indices_idx)
        ]
    )

    # sort indices by their combined similarity score to pick top k
    combined_sorted_result_indices = np.sort(-combined_results_similarity, axis=0)

    # manage history and only keep last 100 results
    new_historical_book_ids_lst = [
        *historical_book_ids_lst.copy(),
        *combined_results_indices_idx.tolist()[:15],
    ][-100:]

    interpretable_search_top_k_df = get_top_n_matching_book_info(
        idx_top_n_np=combined_results_indices,
        sim_score_top_n_np=combined_sorted_result_indices,
        comic_info_dict=book_metadata_dict,
        print_n=top_k,
        query_book_id=query_book_id,
        feature_similarity_type="interpretable_combined",
    )

    return (interpretable_search_top_k_df, new_historical_book_ids_lst)


# def find_best_features_using_triplet_loss(
#     anchors_book_idx,
#     pos_constraint_book_idx,
#     neg_constraint_book_idx,
#     triplet_loss_type="group",
# ):

#     anchor_feature_np = interpretable_scaled_features_np[anchors_book_idx, :]
#     positive_feature_np = interpretable_scaled_features_np[pos_constraint_book_idx, :]
#     negative_feature_np = interpretable_scaled_features_np[neg_constraint_book_idx, :]

#     if triplet_loss_type == "group":
#         feature_importance = triplet_loss.group_triplet_loss(
#             anchor_feature_np=anchor_feature_np,
#             positive_feature_np=positive_feature_np,
#             negative_feature_np=negative_feature_np,
#             global_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
#             fd=[0, 3, 48, 64, 65, 2113, 2117],
#             epochs=500,
#         )
#     else:
#         feature_importance = triplet_loss.individual_triplet_loss(
#             anchor_feature_np=anchor_feature_np,
#             positive_feature_np=positive_feature_np,
#             negative_feature_np=negative_feature_np,
#             individual_global_weights=np.ones(shape=positive_feature_np.shape[1]),
#             cols_lst=interpretable_feature_lst,
#         )

#     return feature_importance


# def find_best_features_using_supervised_feature_importance(selected_idx):

#     features_np = interpretable_scaled_features_np[selected_idx, :]
#     labels_np = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

#     (
#         ridge_regression_feature_importance,
#         logistic_regression_feature_importance,
#         sgd_feature_importance,
#     ) = supervised_feature_importance_using_different_models(
#         features_np, labels_np, interpretable_feature_lst
#     )
#     return (
#         ridge_regression_feature_importance,
#         logistic_regression_feature_importance,
#         sgd_feature_importance,
#     )


# def find_best_features_using_unsupervised_feature_importance(selected_idx):
#     features_np = interpretable_scaled_features_np[selected_idx, :]
#     labels_np = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

#     feature_importance = unsupervised_feature_importance_from_laplacian(
#         features_np=features_np,
#         labels_np=labels_np,
#         col_name_lst=interpretable_feature_lst,
#     )
#     return feature_importance


# def comics_coarse_search(
#     query_comic_book_id: int, feature_weight_dict: dict, top_n: int
# ):

#     # remove this later
#     query_book_id = query_comic_book_id  # -3451

#     # get similarity for all features
#     gender_cosine_similarity = utils.cosine_similarity(
#         gender_feat_np[:, :],
#         gender_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
#     )
#     supersense_cosine_similarity = utils.cosine_similarity(
#         supersense_feat_np[:, :],
#         supersense_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
#     )
#     genre_cosine_similarity = utils.cosine_similarity(
#         genre_feat_np[:, :], genre_feat_np[max(query_book_id, 0) : query_book_id + 1, :]
#     )
#     panel_ratio_cosine_similarity = utils.cosine_similarity(
#         panel_ratio_feat_np[:, :],
#         panel_ratio_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
#     )

#     # print(gender_cosine_similarity.shape,supersense_cosine_similarity.shape, genre_cosine_similarity.shape,  panel_ratio_cosine_similarity.shape)
#     # combine similarity and weigh them
#     combined_results_similarity = (
#         gender_cosine_similarity * feature_weight_dict["gender"]
#         + supersense_cosine_similarity * feature_weight_dict["super_sense"]
#         + genre_cosine_similarity * feature_weight_dict["genre"]
#         + panel_ratio_cosine_similarity * feature_weight_dict["story_pace"]
#     )
#     print(combined_results_similarity.shape)
#     # find top book indices according to combined similarity
#     combined_results_indices = np.argsort(
#         np.squeeze(-combined_results_similarity), axis=0
#     )

#     # sort indices by their combined similarity score to pick top k
#     combined_sorted_result_indices = np.sort(-combined_results_similarity, axis=0)

#     top_k_df = get_top_n_matching_book_info(
#         idx_top_n_np=combined_results_indices,
#         sim_score_top_n_np=combined_sorted_result_indices,
#         comic_info_dict=book_metadata_dict,
#         print_n=top_n,
#         query_book_id=query_book_id,
#         feature_similarity_type="interpretable_combined",
#     )

#     return top_k_df


if __name__ == "__main__":
    # anchors_book_idx = [549] # [79] # # [530] #[34]
    # pos_constraint_book_idx =  [542, 544, 546 ]# [47, 72, 80] # [533, 545, 670] #[14, 40, 99]
    # neg_constraint_book_idx =  [580, 770, 880] # [770, 128, 148] # [710, 760, 800] #[1, 20, 56]

    anchors_book_idx = [552]  # # [530] #[34]
    pos_constraint_book_idx = [537, 539, 533]  # [533, 545, 670] #[14, 40, 99]
    neg_constraint_book_idx = [770, 128, 148]  # [710, 760, 800] #[1, 20, 56]

    top_n = 21
    users_idx = anchors_book_idx + pos_constraint_book_idx + neg_constraint_book_idx
    selected_idx = users_idx  # np.random.choice(idx_arr, 7, replace=False)
    # print(
    #     "Anchor: {}  | Positive: {} | Negative: {}".format(
    #         selected_idx[0], selected_idx[1:4], selected_idx[4:]
    #     )
    # )

    feature_importance = find_best_features_using_triplet_loss(
        anchors_book_idx=anchors_book_idx,
        pos_constraint_book_idx=pos_constraint_book_idx,
        neg_constraint_book_idx=neg_constraint_book_idx,
        selected_idx=selected_idx,
    )
    print()
    print("Group Based triplet loss: {}".format(feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ ==")
    print()

    top_n_results_df = comics_coarse_search(
        query_comic_book_id=anchors_book_idx[0],
        feature_weight_dict=feature_importance,
        top_n=top_n,
    )
    print(top_n_results_df.head(top_n))

    """
    feature_importance = find_best_features_using_triplet_loss(
        anchors_book_idx=anchors_book_idx,
        pos_constraint_book_idx=pos_constraint_book_idx,
        neg_constraint_book_idx=neg_constraint_book_idx,
        selected_idx=selected_idx,
        triplet_loss_type="individual",
    )

    print()
    print("Individual Based triplet loss: {}".format(feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ ==")
    print()
    """

    (
        ridge_regression_feature_importance,
        logistic_regression_feature_importance,
        sgd_feature_importance,
    ) = find_best_features_using_supervised_feature_importance(
        selected_idx=selected_idx
    )
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()
    print(
        "ridge regression feature importance: {}".format(
            ridge_regression_feature_importance
        )
    )
    print()
    print(
        "logistic regression feature importance: {}".format(
            logistic_regression_feature_importance
        )
    )
    print()
    print("sgd feature importance: {}".format(sgd_feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()

    laplacian_feature_importance = find_best_features_using_unsupervised_feature_importance(
        selected_idx=selected_idx
    )
    print("laplacian feature importance: {}".format(laplacian_feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()

