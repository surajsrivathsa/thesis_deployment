from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
import pandas as pd, numpy as np, os, sys
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import itertools
import lime
import lime.lime_tabular
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst
from search.interpretable.triplet_loss import group_triplet_loss

# clf = PassiveAggressiveClassifier(max_iter=100, random_state=7, tol=1e-3)
clf_pipe = Pipeline(
    [
        ("scl", StandardScaler()),
        ("features_np", np.zeros((15, 6))),
        ("labels_np", np.zeros((15,))),
        (
            "clf",
            SGDClassifier(
                max_iter=500,
                tol=2e-3,
                loss="modified_huber",
                random_state=7,
                validation_fraction=0.2,
            ),
        ),
    ]
)


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


def normalize_feature_weights(feature_importance_dict: dict, dist_margin=0.1):
    normalized_feature_importance_dict = {}

    feature_importance_lst = list(feature_importance_dict.values())
    max_feat_val = max(feature_importance_lst)
    min_feat_val = min(feature_importance_lst)
    print("min: {}, max: {}".format(min_feat_val, max_feat_val))
    avg_feat_val = sum(feature_importance_lst) / (len(feature_importance_lst) + 0.01)

    for key, val in feature_importance_dict.items():
        normalized_feature_importance_dict[key] = abs(
            val - min_feat_val + dist_margin
        ) / (abs(max_feat_val - min_feat_val) + dist_margin)

        # print()
        # print("numerator: {}".format(abs(val - min_feat_val + dist_margin)))
        # print("denominator: {}".format(abs(max_feat_val) + dist_margin))
        # print()

    return normalized_feature_importance_dict


def partial_pipe_fit(
    pipeline_obj, features_np, class_np, classes=["not_interested", "interested"]
):
    X_scaled = pipeline_obj.named_steps["scl"].fit_transform(features_np)
    Y = class_np.ravel()
    pipeline_obj.named_steps["features_np"] = X_scaled
    pipeline_obj.named_steps["labels_np"] = Y
    class_weights = compute_class_weight(
        class_weight="balanced", y=Y, classes=np.unique(Y),
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=Y)
    print("class weights: {}".format(class_weights))
    print("sample weights: {}".format(sample_weight))
    pipeline_obj.named_steps["clf"] = pipeline_obj.named_steps["clf"].partial_fit(
        X_scaled, Y, classes=np.unique(Y), sample_weight=sample_weight
    )
    return pipeline_obj


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


def permutation_based_feature_importance(
    model, X, y, stddev_weight=2, feature_col_labels_lst=[]
):
    X_scaled = model.named_steps["scl"].fit_transform(X)
    r = permutation_importance(
        model.named_steps["clf"], X_scaled, y, n_repeats=30, random_state=5
    )
    feature_importance_dict = {}
    mean_lst = []
    feature_nm_lst = []

    # for rank, i in enumerate(r.importances_mean.argsort()[::-1]):
    # print(r.importances_mean)
    for rank, i in enumerate(r.importances_mean.argsort()[::-1]):
        # if r.importances_mean[i] - stddev_weight * r.importances_std[i] > 0:
        # print("Feat Name: {}  : {} +/- {}".format(feature_col_labels_lst[i], r.importances_mean[i], r.importances_std[i],))
        feature_importance_dict[feature_col_labels_lst[i]] = [
            r.importances_mean[i],
            r.importances_std[i],
            rank,
        ]
        mean_lst.append(r.importances_mean[i])
        feature_nm_lst.append(feature_col_labels_lst[i])

    normalized_feature_importance_lst = [
        (float(mn) - min(mean_lst) + 1e-3) / (max(mean_lst) - min(mean_lst) + 1e-2)
        for mn in mean_lst
    ]

    normalized_feature_importance_dict = {
        feat_name: norm_mn
        for feat_name, norm_mn in zip(feature_nm_lst, normalized_feature_importance_lst)
    }

    # print(feature_importance_dict)

    return (feature_importance_dict, normalized_feature_importance_dict)


def lime_based_feature_importance(
    clf_pipe,
    X,
    Y,
    feature_col_labels_lst=[
        "gender",
        "supersense",
        "genre_comb",
        "panel_ratio",
        "comic_cover_img",
        "comic_cover_txt",
    ],
):

    X_scaled = clf_pipe.named_steps["scl"].fit_transform(X)

    # create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_col_labels_lst,
        class_names=["not_interested", "interested"],
        mode="classification",
        random_state=42,
    )

    # select a 10 random sample of instances to explain
    sample_indices = np.random.choice(X_scaled.shape[0], size=12, replace=False)
    sample_instances = X_scaled[sample_indices]
    sample_labels = Y[sample_indices]

    # generate explanations for each instance
    explanations = []
    for i in range(len(sample_instances)):
        exp = explainer.explain_instance(
            data_row=sample_instances[i],
            predict_fn=clf_pipe.named_steps["clf"].predict_proba,
            num_features=6,
            labels=(1,),
        )
        explanations.append(exp.as_list(label=1))

    print()
    print(" ----------- -------------- ------------- ------------ ")
    print()
    print("Initial Explanations: {}".format(explanations))
    print(explainer.feature_names)

    # combine explanations
    combined_exp = {}
    for feature_name in explainer.feature_names:
        combined_exp[feature_name] = 0.0

    for exp in explanations:
        print(combined_exp)
        for feature_name, feature_weight in exp:
            feature_name = [
                x for x in feature_name.split(" ") if x in explainer.feature_names
            ][0]

            combined_exp[feature_name] += float(feature_weight)
            print(feature_name, combined_exp[feature_name])

    # normalize combined explanation
    total_weight = sum(abs(weight) for weight in combined_exp.values())

    print("Combined Lime Explanations: ".format(combined_exp))
    print("Total Lime Explanations: ".format(total_weight))
    for feature_name in explainer.feature_names:
        combined_exp[feature_name] /= total_weight + 1e-3

    print("Normalized Lime Explanations: ".format(combined_exp))
    print()
    print(" ----------- -------------- ------------- ------------ ")
    print()


def adapt_facet_weights_from_previous_timestep_click_info(
    previous_click_info_lst: list, query_book_id: int
):

    selected_idx = [
        d["comic_no"] for d in previous_click_info_lst if d["comic_no"] != query_book_id
    ]
    # print(selected_idx)
    previous_labels_lst = [
        d["interested"]
        for d in previous_click_info_lst
        if d["comic_no"] != query_book_id
    ]  # [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    # query_book_id = query_book_id  # [d["comic_no"] for d in previous_click_info_lst if d["is_query"] == 1][0]

    features_np = interpretable_scaled_features_np[selected_idx, :]
    labels_np = np.array([previous_labels_lst]).T

    # featurize the facets with query combo
    gender_l1_feat_np = utils.l1_similarity(
        gender_feat_np[selected_idx, :],
        gender_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    supersense_l1_feat_np = utils.l1_similarity(
        supersense_feat_np[selected_idx, :],
        supersense_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    genre_l1_feat_np = utils.l1_similarity(
        genre_feat_np[selected_idx, :],
        genre_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    panel_l1_feat_np = utils.l1_similarity(
        panel_ratio_feat_np[selected_idx, :],
        panel_ratio_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    comic_cover_img_l1_feat_np = utils.l1_similarity(
        comic_cover_img_np[selected_idx, :],
        comic_cover_img_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    comic_cover_txt_l1_feat_np = utils.l1_similarity(
        comic_cover_txt_np[selected_idx, :],
        comic_cover_txt_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    features_np = np.zeros((len(selected_idx), 6))
    features_np[:, 0] = gender_l1_feat_np
    features_np[:, 1] = supersense_l1_feat_np
    features_np[:, 2] = genre_l1_feat_np
    features_np[:, 3] = panel_l1_feat_np
    features_np[:, 4] = comic_cover_img_l1_feat_np
    features_np[:, 5] = comic_cover_txt_l1_feat_np

    # retrieve global variable
    global clf_pipe
    clf_pipe = partial_pipe_fit(clf_pipe, features_np, labels_np)

    # find lime feature importance
    # lime_based_feature_importance(
    #     clf_pipe=clf_pipe,
    #     X=features_np,
    #     Y=labels_np,
    #     feature_col_labels_lst=[
    #         "gender",
    #         "supersense",
    #         "genre_comb",
    #         "panel_ratio",
    #         "comic_cover_img",
    #         "comic_cover_txt",
    #     ],
    # )

    # find pfi based feature importance
    (
        feature_importance_dict,
        normalized_feature_importance_dict,
    ) = permutation_based_feature_importance(
        model=clf_pipe,
        X=features_np,
        y=labels_np,
        stddev_weight=0.5,
        feature_col_labels_lst=[
            "gender",
            "supersense",
            "genre_comb",
            "panel_ratio",
            "comic_cover_img",
            "comic_cover_txt",
        ],
    )

    print()
    print(" PFI based feature importance ")
    print(feature_importance_dict, normalized_feature_importance_dict)
    print()

    return (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_pipe["clf"].coef_,
    )


def adapt_facet_weights_from_previous_timestep_click_info_triplet_loss(
    previous_click_info_lst: list, query_book_id: int
):
    """
        case 1:
        len(interested_books) > 0 and len(interested_books) > 0:
            - use itertools.product to get all possible combinations of anchor, positive and negative
            - if number of combinations are more than 100, then select randomly 100
            - form three sets of numy arrays one each for anchor, positive and negative
            - try calling group triplet loss, if it fails use default feature importance.
        case 2:
        len(interested_books) > 0 and len(interested_books) == 0:
            - keep previous feature importance values
        case 3:
        len(interested_books) == 0 and len(interested_books) > 0:
            - use default feature importance values
        case 4:
        len(interested_books) == 0 and len(interested_books) == 0:
            - default feature importance values
    """

    # declare default feature importance incase of any failure
    default_feature_importance = {
        "gender": 1.0,
        "supersense": 1.0,
        "genre_comb": 1.0,
        "panel_ratio": 1.0,
        "comic_cover_img": 1.0,
        "comic_cover_txt": 1.0,
    }

    # separate out positive and negative books
    all_books_except_query_idx_lst = [
        d["comic_no"] for d in previous_click_info_lst if d["comic_no"] != query_book_id
    ]

    # positive books are interested books which were hovered => p
    interested_books_except_query_idx_lst = [
        d["comic_no"]
        for d in previous_click_info_lst
        if d["comic_no"] != query_book_id and d["interested"] > 0.1
    ]

    # negative books were non-hovered disinterested books. => n
    discarded_books_except_query_idx_lst = [
        d["comic_no"]
        for d in previous_click_info_lst
        if d["comic_no"] != query_book_id and d["interested"] < 0.1
    ]

    # anchor
    anchor_idx_lst = [query_book_id]

    all_books_lst = [
        anchor_idx_lst,
        interested_books_except_query_idx_lst,
        discarded_books_except_query_idx_lst,
    ]

    if (
        len(interested_books_except_query_idx_lst) > 0
        and len(discarded_books_except_query_idx_lst) > 0
    ):
        
        # Use semi-hard triplet sampling
        semi_hard_triplets = semi_hard_triplet_sampling(query_book_id, interested_books_except_query_idx_lst, discarded_books_except_query_idx_lst)
        
        if semi_hard_triplets:
            combination_tuple_lst = semi_hard_triplets
        else:
            # Fallback to existing strategy if no semi-hard triplets found
            combination_tuple_lst = [p for p in itertools.product(*all_books_lst)]

        # combination_tuple_lst = [p for p in itertools.product(*all_books_lst)]
        # [(5, 100, 9), (5, 100, 2), (5, 101, 2)]

        anchor_feat_np = np.zeros(
            (len(combination_tuple_lst), interpretable_scaled_features_np.shape[1]),
            dtype="float",
        )
        positive_feat_np = np.zeros(
            (len(combination_tuple_lst), interpretable_scaled_features_np.shape[1]),
            dtype="float",
        )
        negative_feat_np = np.zeros(
            (len(combination_tuple_lst), interpretable_scaled_features_np.shape[1]),
            dtype="float",
        )

        for index, (anchor_idx, positive_idx, negative_idx) in enumerate(
            combination_tuple_lst
        ):
            anchor_feat_np[index, :] = interpretable_scaled_features_np[anchor_idx, :]
            positive_feat_np[index, :] = interpretable_scaled_features_np[
                positive_idx, :
            ]
            negative_feat_np[index, :] = interpretable_scaled_features_np[
                negative_idx, :
            ]

        try:
            feature_importance_dict = group_triplet_loss(
                anchor_feature_np=anchor_feat_np,
                positive_feature_np=positive_feat_np,
                negative_feature_np=negative_feat_np,
                global_weights=np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
                fd=[0, 3, 48, 64, 66, 2114, 2179],
                epochs=501,
                margin=0.2,
                learning_rate=2e-3,
            )
            normalized_feature_importance_dict = normalize_feature_weights(
                feature_importance_dict
            )
            print()
            print(
                "Triplet loss real feature importance: {}".format(
                    feature_importance_dict
                )
            )
            print()
            print(
                " ================= =================== ================== =================== "
            )
            print()
            print(
                "Triplet loss normalized feature importance: {}".format(
                    normalized_feature_importance_dict
                )
            )
            print()
        except Exception as e:
            print("error in triplet loss: {}".format(repr(e)))
            print(traceback.format_exc())
            feature_importance_dict = default_feature_importance.copy()
            normalized_feature_importance_dict = feature_importance_dict.copy()
        finally:
            clf_coeff = None
    else:
        feature_importance_dict = default_feature_importance.copy()
        normalized_feature_importance_dict = feature_importance_dict.copy()
        clf_coeff = None

    return (feature_importance_dict, normalized_feature_importance_dict, clf_coeff)


def semi_hard_triplet_sampling(anchor_idx, positives, negatives, margin=0.2):
    """
    Select semi-hard negative samples for each anchor-positive pair.
    A negative sample is considered semi-hard if it is further away from the anchor 
    than the positive, but within a margin.
    """
    triplets = []
    anchor_vector = get_feature_vector_by_book_id(anchor_idx)

    for positive_idx in positives:
        positive_vector = get_feature_vector_by_book_id(positive_idx)
        positive_distance = np.linalg.norm(anchor_vector - positive_vector)

        for negative_idx in negatives:
            negative_vector = get_feature_vector_by_book_id(negative_idx)
            negative_distance = np.linalg.norm(anchor_vector - negative_vector)

            if positive_distance < negative_distance < positive_distance + margin:
                triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets

def get_feature_vector_by_book_id(id):
    return interpretable_scaled_features_np[id, :]

if __name__ == "__main__":

    comic_no_lst = [
        336,
        149,
        41,
        79,
        21,
        385,
        111,
        305,
        265,
        78,
        387,
        415,
        154,
        195,
        320,
        318,
        251,
        447,
        462,
        46,
        179,
        450,
        82,
        774,
        4,
        289,
        1518,
        463,
        142,
        1493,
        161,
        418,
        80,
        457,
        466,
        250,
        1452,
        1233,
        54,
        426,
        356,
        301,
        378,
        37,
        634,
        134,
        1341,
        471,
        467,
        75,
        1504,
        390,
        465,
        256,
        20,
        1508,
        1562,
        504,
        641,
        330,
        177,
        123,
        1117,
        636,
        182,
        339,
        225,
        113,
        279,
        643,
        249,
        340,
        1132,
        83,
        496,
        136,
        324,
        1519,
        1065,
        487,
        129,
        335,
        449,
        1489,
        288,
        285,
        56,
        1503,
        515,
        446,
        421,
        282,
        581,
        261,
        1337,
        1066,
        211,
        456,
        802,
        688,
        263,
        72,
        323,
        1555,
        185,
        1116,
        226,
        1560,
        24,
        291,
        483,
        281,
        10,
        939,
        247,
        485,
        1514,
        139,
        223,
        1063,
        412,
        362,
        183,
        303,
        1565,
        513,
        30,
        1535,
        1137,
        431,
        255,
        153,
        484,
        1018,
        68,
        42,
        518,
        36,
        205,
        1386,
        198,
        906,
        1463,
        904,
        232,
        107,
        203,
        14,
        1506,
        1062,
        1045,
        23,
        128,
        81,
        109,
        1494,
        187,
        687,
        384,
        377,
        99,
        901,
        1520,
        480,
        1134,
        1136,
        114,
        224,
        199,
        193,
        16,
        521,
        357,
        1135,
        1650,
        1026,
        1479,
        420,
        1509,
        1563,
        380,
        284,
        1390,
        619,
        509,
        150,
        647,
        287,
        936,
        220,
        63,
        244,
        981,
        455,
        1068,
        652,
        334,
        1564,
        974,
    ]
    clicked_lst = [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    previous_click_info_lst = [
        {"comic_no": comic_no, "clicked": clicked}
        for comic_no, clicked in zip(comic_no_lst, clicked_lst)
    ]

    """
    previous_click_info_lst = [
        {"comic_no": 537, "clicked": 1.0},
        {"comic_no": 539, "clicked": 1.0},
        {"comic_no": 553, "clicked": 0.0},
        {"comic_no": 770, "clicked": 0.0},
        {"comic_no": 1144, "clicked": 0.0},
        {"comic_no": 1026, "clicked": 0.0},
        {"comic_no": 1, "clicked": 0.0},
        {"comic_no": 128, "clicked": 0.0},
        {"comic_no": 148, "clicked": 0.0},
        {"comic_no": 533, "clicked": 1.0},
        {"comic_no": 334, "clicked": 1.0},
        {"comic_no": 150, "clicked": 0.0},
        {"comic_no": 700, "clicked": 0.0},
        {"comic_no": 750, "clicked": 0.0},
        {"comic_no": 800, "clicked": 0.0},
        {"comic_no": 1200, "clicked": 0.0},
        {"comic_no": 652, "clicked": 0.0},
        {"comic_no": 1558, "clicked": 0.0},
        {"comic_no": 1500, "clicked": 0.0},
        {"comic_no": 1005, "clicked": 0.0},
        {"comic_no": 1234, "clicked": 0.0},
    ]
    """
    (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_coeff,
    ) = adapt_facet_weights_from_previous_timestep_click_info(
        previous_click_info_lst, query_book_id=5
    )

    print()
    print(" =================== =================== =================== ")
    print()
    print(normalized_feature_importance_dict)
    print()
    print(" =================== =================== =================== ")
    print()
    print(feature_importance_dict)
    print()
    print(" =================== =================== =================== ")
    print()
    print(clf_coeff)

