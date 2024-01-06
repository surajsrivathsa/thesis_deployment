import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import uuid
import datetime
import os
import json

## import custom functions
import common_constants.backend_constants as cst


def cosine_similarity(u: np.array, v: np.array):
    u = np.expand_dims(u, 1)
    n = np.sum(u * v, axis=2)
    # print()
    # print(u.shape, n.shape)
    # print()
    # print(np.linalg.norm(u, axis=2).shape, np.linalg.norm(v, axis=1).shape)
    # print()
    d = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=1)
    return n / (d + 1e-6)


def l2_similarity(u: np.array, v: np.array):
    d = np.linalg.norm(u - v, axis=1)
    return d


def l1_similarity(u: np.array, v: np.array):
    d = np.linalg.norm(u - v, ord=1, axis=1)
    return d


def load_book_metadata():
    comic_book_filepath = cst.BOOK_METADATA_FILEPATH
    comic_book_metadata_df = pd.read_csv(comic_book_filepath)
    comic_book_metadata_df["our_idx"] = comic_book_metadata_df.index.copy()

    book_metadata_dict = {}
    counter = 0
    for idx, row in comic_book_metadata_df.iterrows():
        if str(row["Book Title"]) != "nan":
            book_metadata_dict[counter] = [
                row["comic_no"],
                row["Book Title"],
                row["genre"],
                row["Year"],
            ]
            counter += 1

    print(book_metadata_dict[0])
    return (book_metadata_dict, comic_book_metadata_df)


def load_all_coarse_features():
    cld_tfidf_df = pd.read_csv(cst.CLD_TF_IDF_FILEPATH)
    cld_tfidf_np = cld_tfidf_df.to_numpy()
    cld_tfidf_np = cld_tfidf_np[:, 1:]

    edh_tfidf_df = pd.read_csv(cst.EDH_TF_IDF_FILEPATH)
    edh_tfidf_np = edh_tfidf_df.to_numpy()
    edh_tfidf_np = edh_tfidf_np[:, 1:]

    hog_tfidf_df = pd.read_csv(cst.HOG_TF_IDF_FILEPATH)
    hog_tfidf_np = hog_tfidf_df.to_numpy()
    hog_tfidf_np = hog_tfidf_np[:, 1:]

    text_tfidf_df = pd.read_csv(cst.TEXT_TF_IDF_FILEPATH)
    text_tfidf_np = text_tfidf_df.to_numpy()
    text_tfidf_np = text_tfidf_np[:, 1:]

    comic_cover_img_df = pd.read_csv(cst.COMIC_COVER_IMG_FILEPATH)
    comic_cover_img_np = comic_cover_img_df.to_numpy()
    comic_cover_img_np = comic_cover_img_np[:, 1:]

    comic_cover_txt_df = pd.read_csv(cst.COMIC_COVER_TXT_FILEPATH)
    comic_cover_txt_np = comic_cover_txt_df.to_numpy()
    comic_cover_txt_np = comic_cover_txt_np[:, 1:]

    return (
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
    )


def load_all_interpretable_features_old():

    with open(cst.STORY_PACE_FEATURE_FILEPATH, "rb") as handle:
        pace_of_story_info_feature_dict = pickle.load(handle)

    panel_ratio_np = np.array(
        pace_of_story_info_feature_dict["panel_ratio_per_book"][:165]
    )
    panel_ratio_np = (panel_ratio_np - np.min(panel_ratio_np) + 1e-6) / (
        np.max(panel_ratio_np) + 1e-6 - np.min(panel_ratio_np)
    )
    print(panel_ratio_np.shape)
    panel_ratio_lst = panel_ratio_np.tolist()

    interpretable_features_df = pd.read_csv(cst.INTERPRETABLE_FEATURES_FILEPATH)
    interpretable_features_df["panel_ratio"] = panel_ratio_lst
    interpretable_features_df = interpretable_features_df.iloc[:165, 2:]
    interpretable_feature_lst = list(interpretable_features_df.columns)
    interpretable_features_np = interpretable_features_df.to_numpy()

    interpretable_scaled_features_np = MinMaxScaler().fit_transform(
        interpretable_features_df.values
    )

    return (
        interpretable_scaled_features_np,
        interpretable_feature_lst,
    )


def load_all_interpretable_features():
    interpretable_features_df = pd.read_csv(cst.INTERPRETABLE_FEATURES_FILEPATH)
    interpretable_features_df = interpretable_features_df.iloc[:, 1:]
    interpretable_feature_lst = list(interpretable_features_df.columns)
    interpretable_scaled_features_np = (
        interpretable_features_df.to_numpy()
    )  # MinMaxScaler().fit_transform(interpretable_features_df.values)

    print(
        "interpretable features metadata: {}".format(
            interpretable_scaled_features_np.shape
        )
    )
    # print('interpretable features used: {}'.format(interpretable_feature_lst))

    # get individual interpretable features
    gender_feat_np = interpretable_scaled_features_np[:, 0:3].copy()
    supersense_feat_np = interpretable_scaled_features_np[:, 3:48].copy()
    genre_feat_np = interpretable_scaled_features_np[:, 48:64].copy()
    panel_ratio_feat_np = interpretable_scaled_features_np[:, 64:66].copy()

    comic_cover_img_df = pd.read_csv(cst.COMIC_COVER_IMG_FILEPATH)
    comic_cover_img_np = comic_cover_img_df.to_numpy()
    comic_cover_img_np = comic_cover_img_np[:, 1:]

    comic_cover_txt_df = pd.read_csv(cst.COMIC_COVER_TXT_FILEPATH)
    comic_cover_txt_np = comic_cover_txt_df.to_numpy()
    comic_cover_txt_np = comic_cover_txt_np[:, 1:]

    return (
        interpretable_scaled_features_np,
        interpretable_feature_lst,
        gender_feat_np,
        supersense_feat_np,
        genre_feat_np,
        panel_ratio_feat_np,
        comic_cover_img_np,
        comic_cover_txt_np,
    )


def load_local_explanation_story_pace():

    with open(cst.STORY_PACE_FEATURE_FILEPATH, "rb") as handle:
        story_pace_feature_dict = pickle.load(handle)

    return story_pace_feature_dict


def load_local_explanation_book_cover():

    with open(cst.BOOK_COVER_PROMPT_FILEPATH, "rb") as handle:
        book_cover_prompt_dict = pickle.load(handle)

    return book_cover_prompt_dict


def load_local_explanation_w5_h1_facets():
    w5_h1_dict = {}

    with open(cst.W5_H1_FACETS_JSON_FILEPATH) as f:
        spellchecked_parsed_json_lst = json.load(f)
        for valdict in spellchecked_parsed_json_lst:
            w5_h1_dict[valdict["comic_no"]] = valdict

    print(w5_h1_dict[540])

    return w5_h1_dict


def load_local_explanation_characters():
    characters_info_dict = {}
    tmp_characters_info_dict = {}

    with open(cst.CHARACTERS_JSON_FILEPATH) as f:
        tmp_characters_info_dict = json.load(f)

    for key, val in tmp_characters_info_dict.items():
        characters_info_dict[int(key)] = val

    print(characters_info_dict[540])

    return characters_info_dict


def filter_fnc(doc):
    if (
        "comic book cover" not in doc
        and "blue-ray" not in doc
        and "ebay" not in doc
        and "bbc" not in doc
        and "comic book sitting" not in doc
        and "cover of a comic book" not in doc
        and "metal plate photograph" not in doc
        and "rounded corners" not in doc
        and "super-resolution" not in doc
        and "tabloid" not in doc
        and "border" not in doc
        and "smooth panelling" not in doc
        and "one panel" not in doc
        and "bunch of comics" not in doc
        and "image of a comic book page" not in doc
        and "book with pictures on it" not in doc
        and "high resolution product photo" not in doc
        and "low quality photo" not in doc
        and "full colour print" not in doc
        and "by " not in doc
        and "cover image" not in doc
        and "colorised" not in doc
        and "full-color" not in doc
        and "art style" not in doc
        and "paperback cover" not in doc
        and "retro cover" not in doc
        and "author" not in doc
        and "grainy image" not in doc
        and "cover of a magazine" not in doc
        and "electronic ads" not in doc
        and "middle of the page" not in doc
        and "then another" not in doc
        and "a picture of a comic strip in a frame" not in doc
        and "full page scan" not in doc
        and "super - resolution" not in doc
        and "grainy" not in doc
        and "listing image" not in doc
        and "dialog text" not in doc
        and "in the original box" not in doc
        and "playboy cover" not in doc
        and "yellowed paper" not in doc
        and "screenshot" not in doc
        and "promotional render" not in doc
        and "full - color" not in doc
        and "blue - ray" not in doc
        and "a picture of a picture of a comic strip" not in doc
        and "museum catalog photograph" not in doc
        and "professional high quality scan" not in doc
        and "weather report" not in doc
        and "copyright" not in doc
        and "magazine" not in doc
        and "product" not in doc
        and "highly detailed form" not in doc
        and "flash on camera" not in doc
        and "commercial banner" not in doc
        and "camera flash" not in doc
        and "old footage" not in doc
        and "textbook page" not in doc
        and "comic book black lines" not in doc
        and "text paragraphs in left" not in doc
        and "meme template" not in doc
        and "manga panel" not in doc
        and "with highly detailed" not in doc
        and "lower quality" not in doc
        and "tin foiling" not in doc
        and "blue - ray screenshot" not in doc
        and "full page" not in doc
        and "reduce duplication" not in doc
        and "very stylized" not in doc
        and "circa 1" not in doc
        and "sfx" not in doc
        and "2 d animation" not in doc
        and "part of the screen" not in doc
        and "60s poster" not in doc
        and "color displacement" not in doc
        and "screen shot" not in doc
        and "entire character!" not in doc
        and "bottom shot" not in doc
        and "marvel 8 0" not in doc
        and "marvel 1" not in doc
        and ":" not in doc
        and "edition" not in doc
        and "editorial" not in doc
        and "full color still" not in doc
        and "newspaper" not in doc
        and "advertisement" not in doc
        and "super complex" not in doc
        and "various refine" not in doc
        and "several character design" not in doc
        and "1 9 6 0" not in doc
        and "detail shots" not in doc
        and "1 9 3 0" not in doc
        and "1 9 4 0" not in doc
        and "1 9 5 0" not in doc
        and "1 9 7 0" not in doc
        and "general human form" not in doc
        and "visual fidelity" not in doc
        and "comic book panels" not in doc
        and "an action pose" not in doc
        and "4 0" not in doc
        and "5 0" not in doc
        and "6 0" not in doc
        and "7 0" not in doc
        and "8 0" not in doc
        and "multiple stories" not in doc
        and "astounding stories" not in doc
        and "comic page" not in doc
        and "storybook" not in doc
        and "layout design" not in doc
        and "sunday" not in doc
        and "comic book page" not in doc
        and "flash image" not in doc
        and "svg comic style" not in doc
        and "comics style" not in doc
        and "iconic logo" not in doc
        and "a comic strip with an image of a group of people" not in doc
        and "bunch of cartoon characters" not in doc
        and "comic lines" not in doc
        and " composition" not in doc
        and "bunch of people" not in doc
        and "game card" not in doc
        and "font" not in doc
        and "game logo" not in doc
        and "official marvel media" not in doc
        and "without text" not in doc
        and "legible" not in doc
        and "close up of a comic book" not in doc
        and "panel of a marvel comic" not in doc
        and "golden age illustrat" not in doc
        and "dvd cover" not in doc
        and "cover illustration" not in doc
        and "classic chrome" not in doc
        and "complete scene" not in doc
        and "loss comic" not in doc
        and "comic artstyle" not in doc
        and len(doc) > 9
        and len(doc.split()) > 1
    ):
        return True
    else:
        return False


def check_if_hovered(clicksinfo_dict: list, b_id: int):
    interested_book_ids_lst = [
        book["comic_no"] for book in clicksinfo_dict if book["interested"] == 1.0
    ] + [b_id]

    if len(set(interested_book_ids_lst)) > 1:
        return True
    else:
        return False


def create_session_data_folder():
    unique_id = str(uuid.uuid4())
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    final_unique_id = now_str + "|" + unique_id
    os.mkdir(os.path.join(cst.SESSIONDATA_PARENT_FILEPATH, final_unique_id))
    return (
        final_unique_id,
        os.path.join(cst.SESSIONDATA_PARENT_FILEPATH, final_unique_id),
    )


def find_latest_session(unique_id: str):
    latest_session_id = sorted(
        os.listdir(cst.SESSIONDATA_PARENT_FILEPATH), key=os.path.getmtime
    )[0]
    latest_session_folderpath = os.path.join(
        cst.SESSIONDATA_PARENT_FILEPATH, latest_session_id
    )
    return (latest_session_id, latest_session_folderpath)


def log_session_data(session_folderpath: str, data: dict):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    with open(
        os.path.join(session_folderpath, "{}.json".format(now_str)), "w"
    ) as json_filehandler:
        json.dump(data, json_filehandler, indent=2)
    return


def check_if_all_books_are_hovered(clicksinfo_dict: list, b_id: int):
    interested_book_ids_lst = [
        book["comic_no"] for book in clicksinfo_dict if book["interested"] == 1.0
    ] + [b_id]

    if len(set(interested_book_ids_lst)) == 15:
        return True
    else:
        return False

