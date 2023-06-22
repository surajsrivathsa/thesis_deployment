import os, sys
from pathlib import Path

curr_dir = d = Path(__file__).resolve().parents[0]
parent_dir = d = Path(__file__).resolve().parents[1]
print(curr_dir, parent_dir)

## Book metadata information
BOOK_METADATA_FILEPATH = os.path.join(
    parent_dir, "features", "new_comic_book_metadata.csv"
)


## information about coarse non interpretable features
TEXT_TF_IDF_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "text_tf_idf_feat.csv"
)
CLD_TF_IDF_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "cld_tf_idf_feat.csv"
)
EDH_TF_IDF_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "edh_tf_idf_feat.csv"
)
HOG_TF_IDF_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "hog_tf_idf_feat.csv"
)
COMIC_COVER_IMG_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "comic_book_cover_img_feat.csv"
)
COMIC_COVER_TXT_FILEPATH = os.path.join(
    parent_dir, "features", "coarse", "comic_book_cover_txt_feat.csv"
)

## information about interpretable features
INTERPRETABLE_FEATURES_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "all_features_combined.csv"
)
STORY_PACE_FEATURE_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "feature_storypace_all.pkl"
)
BOOK_COVER_PROMPT_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "comic_book_cover_prompt_dict.pickle"
)
W5_H1_FACETS_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "facets_all_dict.pickle"
)
W5_H1_FACETS_JSON_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "spellchecked_parsed_json_lst.json"
)
CHARACTERS_JSON_FILEPATH = os.path.join(
    parent_dir, "features", "interpretable", "character_explanations.json"
)
SESSIONDATA_PARENT_FILEPATH = os.path.join(parent_dir, "data", "session_data")
