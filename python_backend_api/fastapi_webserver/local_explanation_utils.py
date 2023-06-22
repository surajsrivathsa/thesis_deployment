import os, sys
import pandas as pd, numpy as np
import ast
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils


story_pace_feature_dict = utils.load_local_explanation_story_pace()
book_cover_prompt_dict = utils.load_local_explanation_book_cover()
w5_h1_facets_dict = utils.load_local_explanation_w5_h1_facets()
character_info_dict = utils.load_local_explanation_characters()
print("keys local explanation: {}".format(len(story_pace_feature_dict.keys())))
print("example local explanation: {}".format(story_pace_feature_dict[10]))
print()
print(" ================ ================ ================== ")
print()


def fetch_story_pace(book_id: int):
    if book_id in story_pace_feature_dict:
        # print(story_pace_feature_dict[book_id], type(story_pace_feature_dict[book_id]))
        try:
            return [100.0 / (int(i) + 1.0) for i in story_pace_feature_dict[book_id]]
        except Exception as e:
            return [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ]
    else:
        return [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]


def find_numpages_and_storypace(book_id: int):
    default_response = {
        "comic_no": book_id,
        "num_pages": 1,
        "num_panels": 1,
        "pace_of_story": 1,
    }
    story_pace_lst = fetch_story_pace(book_id)

    default_response["num_pages"] = len(story_pace_lst)
    default_response["num_panels"] = sum(
        [int(i) for i in story_pace_feature_dict[book_id]]
    )
    default_response["pace_of_story"] = round(np.mean(story_pace_lst), 2)

    return default_response


def fetch_5w_1h_facets(book_id: int):
    """
    {
        "Who": ["george", "all of you", "man of tomorrow"],
        "What": ["comic strip game", "romance novel", "tiny que romney prmmeneet comic"],
        "When": ["hours ice my min", "this trip", "twomin shinamaroosha saga"],
        "Why": ["why man this trip is so good", "conceive such an innate ly human pictorial document"],
        "Where": ["backcountry mountains", "high up in the mountain", "on him and see if he is making any progress"],
        "How": ["never been accomplished", "exercise his ingenuity to the utmost", "comic and glaphappy setup"],
    }
    """
    default_facets = {
        "Who": [""],
        "What": [""],
        "When": [""],
        "Why": [""],
        "Where": [""],
        "How": [""],
    }
    if book_id in w5_h1_facets_dict:
        # print(w5_h1_facets_dict[book_id], type(w5_h1_facets_dict[book_id]))
        book_facet_obj = w5_h1_facets_dict[book_id]
        if isinstance(book_facet_obj, dict):
            return book_facet_obj
        elif (
            "{" in book_facet_obj
            and "}" in book_facet_obj
            and isinstance(book_facet_obj, str)
        ):
            book_facet_obj = book_facet_obj.replace("JSON Response:", "").strip()
            brackets_start_idx = book_facet_obj.find("{")
            brackets_end_idx = book_facet_obj.find("}")
            book_facet_obj = book_facet_obj[brackets_start_idx : brackets_end_idx + 1]
            book_facet_obj_dict = ast.literal_eval(book_facet_obj)
            if isinstance(book_facet_obj_dict, dict):
                return book_facet_obj_dict
            else:
                return default_facets
        else:
            return default_facets

    else:
        return default_facets
    return ""


def pick_facets_for_local_explanation(book_id: int):
    default_facets = {
        "Who": [],
        "What": [],
        "When": [],
        "Why": [],
        "Where": [],
        "How": [],
    }
    if book_id in w5_h1_facets_dict:
        if (
            "Who" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["Who"]) > 0
        ):
            default_facets["Who"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["Who"],
                    min(3, len(w5_h1_facets_dict[book_id]["Who"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        if (
            "What" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["What"]) > 0
        ):
            default_facets["What"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["What"],
                    min(2, len(w5_h1_facets_dict[book_id]["Who"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        if (
            "When" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["When"]) > 0
        ):
            default_facets["When"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["When"],
                    min(2, len(w5_h1_facets_dict[book_id]["When"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        if (
            "Why" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["Why"]) > 0
        ):
            default_facets["Why"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["Why"],
                    min(2, len(w5_h1_facets_dict[book_id]["Why"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        if (
            "Where" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["Where"]) > 0
        ):
            default_facets["Where"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["Where"],
                    min(2, len(w5_h1_facets_dict[book_id]["Where"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        if (
            "How" in w5_h1_facets_dict[book_id]
            and len(w5_h1_facets_dict[book_id]["How"]) > 0
        ):
            default_facets["How"] = [
                x
                for x in random.sample(
                    w5_h1_facets_dict[book_id]["How"],
                    min(2, len(w5_h1_facets_dict[book_id]["How"])),
                )
                if len(x) > 1
                and "keyword" not in x
                and "marvel" not in x
                and "news" not in x
                and "ads" not in x
            ]

        return default_facets
    else:
        return default_facets


def fetch_book_cover_keywords(book_id: int):
    if book_id in book_cover_prompt_dict:
        # print(book_cover_prompt_dict[book_id], type(book_cover_prompt_dict[book_id]))
        return book_cover_prompt_dict[book_id]
    else:
        return ["comic book cover"]


def fetch_character_info_for_local_explanation(book_id: int):
    default_response = {
        "comic_no": book_id,
        "total_characters": 1,
        "male_characters": [],
        "female_characters": [],
        "occupations": [],
    }
    if (
        book_id not in character_info_dict
        or character_info_dict[book_id]["total_characters"] < 3
    ):
        return default_response
    else:
        default_response["total_characters"] = character_info_dict[book_id][
            "total_characters"
        ]
        default_response["male_characters"] = character_info_dict[book_id][
            "male_characters"
        ]
        default_response["female_characters"] = character_info_dict[book_id][
            "female_characters"
        ]
        default_response["occupations"] = character_info_dict[book_id]["occupations"]

        return default_response

