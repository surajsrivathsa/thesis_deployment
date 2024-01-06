import local_explanation_utils as le_utils
import interpretable_search_utils as is_utils
import rerank_results as rrr
import coarse_search_utils as cs_utils
import explain_relevance_feedback as erf
from search.coarse import coarse_search
import common_constants.backend_constants as cst
import common_functions.backend_utils as utils
from fastapi import FastAPI, Query, HTTPException, File, Request
from fastapi.responses import StreamingResponse
from io import BytesIO

from pydantic import BaseModel
from typing import List, Optional
import json
import os
import sys
import random
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


print("returned to main")

print("coarse search")

print("rrr")

print("is_utils")

print("Server loaded")

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create global variable
sentence_transformer_model = None

# create session id, set this by default
session_id = "bd65600d-8669-4903-8a14-af88203add38"
latest_session_id = "bd65600d-8669-4903-8a14-af88203add38"
latest_session_folderpath = os.path.join(
    cst.SESSIONDATA_PARENT_FILEPATH, session_id)

# create global variable to track history of search results
book_search_results_history_lst = []


# define startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global sentence_transformer_model
    sentence_transformer_model = erf.create_model()
    global book_search_results_history_lst
    book_search_results_history_lst = []


@app.on_event("shutdown")
async def shutdown_event():
    global sentence_transformer_model
    erf.shutdown_model(sentence_transformer_model)
    global book_search_results_history_lst
    book_search_results_history_lst = []


class Book(BaseModel):
    id: int
    comic_no: int
    book_title: str
    genre: str
    year: str
    interested: Optional[float] = 0.0
    thumbsUp: Optional[float] = 0.0
    thumbsDown: Optional[float] = 0.0


class BookList(BaseModel):
    interested_book_lst: List[Book]


class SearchBarQuery(BaseModel):
    id: int
    comic_no: int
    book_title: str
    text: str
    type: str


class FacetWeight(BaseModel):
    gender: float = 1.0
    supersense: float = 1.0
    genre_comb: float = 1.0
    panel_ratio: float = 1.0
    comic_cover_img: Optional[float] = 1.0
    comic_cover_txt: Optional[float] = 1.0
    cld: Optional[float] = 0.1
    edh: Optional[float] = 0.1
    hog: Optional[float] = 0.1
    text: Optional[float] = 1.0
    comic_img: Optional[float] = 0.1
    comic_txt: Optional[float] = 0.1


@app.get("/fake_book/{b_id}", status_code=200)
def get_fake_coarse_results(b_id: int):
    coarse_filtered_book_df = comic_book_metadata_df[
        ["comic_no", "book_title", "genre", "year"]
    ].sample(n=20, random_state=b_id)
    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        coarse_filtered_book_new_lst.append(d)

    print("Query Book : {}".format(b_id))
    return coarse_filtered_book_new_lst


# @app.get("/book/{b_id}", status_code=200)
def get_coarse_results(b_id: int):

    # query_book_comic_id = b_id # 1262 # 1647(tin-tin), 520(aquaman), 558(asterix), 587(Avengers), 650(Batman), 1270(Justice Society)
    top_n = 200
    feature_weight_dict = {
        "cld": 0.1,
        "edh": 0.1,
        "hog": 0.1,
        "text": 1.7,
    }

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
    return coarse_filtered_book_new_lst


def create_fake_clicks_for_previous_timestep_data(
    coarse_filtered_book_df: pd.DataFrame,
):

    index_lst = [i for i in range(20)]
    interested_lst_size = random.randint(2, 5)
    interested_books_idx_lst = np.random.choice(
        index_lst, size=interested_lst_size, replace=False
    )
    coarse_filtered_book_df["interested"] = 0.0
    coarse_filtered_book_df.loc[interested_books_idx_lst, "interested"] = 1.0

    interested_book_lst = (
        coarse_filtered_book_df[["comic_no", "interested"]]
        .iloc[:20, :]
        .copy()
        .fillna("")
        .to_dict("records")
    )
    return interested_book_lst


def create_real_clicks_for_previous_timestamp_data(interested_book_lst: List):
    interested_book_dict_lst = [
        {"comic_no": obj.comic_no, "interested": obj.interested}
        for obj in interested_book_lst
    ]
    return interested_book_dict_lst


def fetch_genres(book_id: int):
    genre_lst = book_metadata_dict[book_id][2].split("|")
    return {"comic_no": book_id, "genre": genre_lst}


@app.post("/book_search", status_code=200)
async def search_with_real_clicks(
    cbl: BookList,
    b_id: int = Query(...),
    generate_fake_clicks: bool = Query(default=True),
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=1.0,
        comic_cover_txt=1.0,
    ),
):
    # print(cbl.clicked_book_lst)
    # print(b_id, generate_fake_clicks)
    # b_id: int, clicksinfo_dict: dict, generate_fake_clicks=True
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_coarse_search(b_id=b_id)

    print(coarse_filtered_book_df.head(100))
    print(
        "generate_fake_clicks: {} | {}".format(
            generate_fake_clicks, type(generate_fake_clicks)
        )
    )

    # if you use search results from search bar reset history
    global book_search_results_history_lst

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )

    else:
        clicksinfo_dict = create_real_clicks_for_previous_timestamp_data(
            cbl.interested_book_lst
        )  # [{"comic_no": "1", "clicked": 1.0}, {"comic_no": "3", "clicked": 0.0}]
    print("clicksinfo_dict: {}".format(clicksinfo_dict))

    if not generate_fake_clicks:

        # handle if user directly clikcs on book without hovering and if all books are hovered
        if utils.check_if_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ) and not utils.check_if_all_books_are_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ):
            (
                feature_importance_dict,
                normalized_feature_importance_dict,
                clf_coef,
            ) = rrr.adapt_facet_weights_from_previous_timestep_click_info_triplet_loss(
                previous_click_info_lst=clicksinfo_dict, query_book_id=b_id
            )
        elif not utils.check_if_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ) and utils.check_if_all_books_are_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ):
            normalized_feature_importance_dict = {
                "gender": input_feature_importance_dict.gender,
                "supersense": input_feature_importance_dict.supersense,
                "genre_comb": input_feature_importance_dict.genre_comb,
                "panel_ratio": input_feature_importance_dict.panel_ratio,
                "comic_cover_img": input_feature_importance_dict.comic_cover_img,
                "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
            }
            clf_coef = None
            feature_importance_dict = normalized_feature_importance_dict
        else:
            normalized_feature_importance_dict = {
                "gender": input_feature_importance_dict.gender,
                "supersense": input_feature_importance_dict.supersense,
                "genre_comb": input_feature_importance_dict.genre_comb,
                "panel_ratio": input_feature_importance_dict.panel_ratio,
                "comic_cover_img": input_feature_importance_dict.comic_cover_img,
                "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
            }
        clf_coef = None
        feature_importance_dict = normalized_feature_importance_dict
    else:
        normalized_feature_importance_dict = {
            "gender": input_feature_importance_dict.gender,
            "supersense": input_feature_importance_dict.supersense,
            "genre_comb": input_feature_importance_dict.genre_comb,
            "panel_ratio": input_feature_importance_dict.panel_ratio,
            "comic_cover_img": input_feature_importance_dict.comic_cover_img,
            "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
        }
        clf_coef = None
        feature_importance_dict = normalized_feature_importance_dict

    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )

    # new_normalized_feature_importance_dict = {"gender": 1.0, "supersense": 1.0, "genre_comb": 1.0, "panel_ratio": 1.0, "comic_cover_img": 1.0, "comic_cover_txt": 1.0}
    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
        historical_book_ids_lst,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
        historical_book_ids_lst=book_search_results_history_lst.copy(),
    )

    book_search_results_history_lst = historical_book_ids_lst.copy()

    if generate_fake_clicks:
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=[],
            query_book_id=b_id,
            search_results=interpretable_filtered_book_lst,
            model=sentence_transformer_model,
        )
    else:

        print("clicksinfo_dict: {}".format(clicksinfo_dict))
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=clicksinfo_dict,
            query_book_id=b_id,
            search_results=interpretable_filtered_book_lst,
            model=sentence_transformer_model,
        )

    # add facet weights to UI dict
    print(
        {
            **interpretable_filtered_book_lst[0],
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
    )
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(interpretable_filtered_book_lst)
        if idx <= 20
    ]

    print(feature_importance_dict, clf_coef)
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    print()
    print(" +++++++++++++ ++++++++++++ +++++++++++++ ")
    print()

    for x in interpretable_filtered_book_lst:
        print(x)

    print()
    print(" +++++++++++++ ++++++++++++ ++++++++++++++ ")
    print()

    # log userrs interaction data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "cbl": cbl.dict(),
                "b_id": b_id,
                "generate_fake_clicks": generate_fake_clicks,
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "search_with_real_clicks",
        },
    )
    return interpretable_filtered_book_new_lst


@app.post("/book_search_with_searchbar_inputs", status_code=200)
async def search_with_searchbar_inputs(
    searchbar_query: SearchBarQuery,
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=1.0,
        comic_cover_txt=1.0,
    ),
):
    print("searchbar_query : {}".format(searchbar_query))

    # if you use search results from search bar reset history
    global book_search_results_history_lst
    book_search_results_history_lst = []

    if (
        searchbar_query.type == "book"
        or searchbar_query.type == "character"
        or searchbar_query.type == "genre"
    ):
        b_id = searchbar_query.comic_no
        (
            coarse_filtered_book_new_lst,
            coarse_filtered_book_df,
        ) = cs_utils.perform_coarse_search(b_id=b_id)
    else:
        b_id = 1
        (
            coarse_filtered_book_new_lst,
            coarse_filtered_book_df,
        ) = cs_utils.perform_coarse_search(b_id=b_id)

    normalized_feature_importance_dict = {
        "gender": input_feature_importance_dict.gender,
        "supersense": input_feature_importance_dict.supersense,
        "genre_comb": input_feature_importance_dict.genre_comb,
        "panel_ratio": input_feature_importance_dict.panel_ratio,
        "comic_cover_img": input_feature_importance_dict.comic_cover_img,
        "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
    }

    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )
    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
        historical_book_ids_lst,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
        historical_book_ids_lst=book_search_results_history_lst.copy(),
    )
    book_search_results_history_lst = historical_book_ids_lst.copy()

    relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
        clicksinfo_dict=[],
        query_book_id=b_id,
        search_results=interpretable_filtered_book_lst,
        model=sentence_transformer_model,
    )

    # add facet weights to UI
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(interpretable_filtered_book_lst)
        if idx <= 20
    ]
    print(
        {**interpretable_filtered_book_lst[0], **normalized_feature_importance_dict})
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    # log data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "searchbar_query": searchbar_query.dict(),
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "search_with_searchbar_inputs",
        },
    )

    return interpretable_filtered_book_new_lst


@app.post("/local_explanation", status_code=200)
async def get_local_explanation(selected_book_lst: dict):

    print(selected_book_lst)
    selected_book_id_1 = selected_book_lst["selected_book_lst"][0]["comic_no"]
    selected_book_id_2 = selected_book_lst["selected_book_lst"][1]["comic_no"]
    print(selected_book_id_1, selected_book_id_2)
    story_pace_book_1 = le_utils.fetch_story_pace(selected_book_id_1)
    story_pace_book_2 = le_utils.fetch_story_pace(selected_book_id_2)

    w5_h1_facets_book_1 = le_utils.pick_facets_for_local_explanation(
        selected_book_id_1)
    w5_h1_facets_book_2 = le_utils.pick_facets_for_local_explanation(
        selected_book_id_2)

    lrp_book_1 = le_utils.fetch_book_cover_keywords(selected_book_id_1)
    lrp_book_2 = le_utils.fetch_book_cover_keywords(selected_book_id_2)
    print(
        "{} | {} | {} | {} | {} | {}".format(
            story_pace_book_1,
            story_pace_book_2,
            w5_h1_facets_book_1,
            w5_h1_facets_book_2,
            lrp_book_1,
            lrp_book_2,
        )
    )

    # log data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {"selected_book_lst": selected_book_lst},
            "output_data": {
                "local_explanations": {
                    "story_pace": [story_pace_book_1, story_pace_book_2],
                    "w5_h1_facets": [w5_h1_facets_book_1, w5_h1_facets_book_2],
                    "lrp_genre": [lrp_book_1, lrp_book_2],
                }
            },
            "function_name": "get_local_explanation",
        },
    )

    return {
        "story_pace": [story_pace_book_1, story_pace_book_2],
        "w5_h1_facets": [w5_h1_facets_book_1, w5_h1_facets_book_2],
        "lrp_genre": [lrp_book_1, lrp_book_2],
    }


@app.get("/start_session", status_code=200)
async def start_session(flag: str):
    print(flag)
    if flag == "startnewsession":
        new_session_id, curr_session_folderpath = utils.create_session_data_folder()
        global session_id
        session_id = new_session_id

        global latest_session_folderpath
        latest_session_folderpath = curr_session_folderpath

    # log data for evaluation
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {"session_id": latest_session_id},
            "output_data": {},
            "function_name": "start_session",
        },
    )
    return {"session_id": session_id}


@app.get(
    "/view_comic_book/{b_id}",
    status_code=200,
    responses={200: {"content": {"application/pdf": {}}}},
    response_class=StreamingResponse,
)
async def view_comic_book(b_id: int = 1):

    pdf_folderpath = "../data/comics_data/comic_books"
    pdf_filepath = os.path.join(
        pdf_folderpath, "comic_book_{}.pdf".format(b_id))
    print("pdf_filepath: {}".format(pdf_filepath))
    if os.path.exists(pdf_filepath):
        new_pdf_filepath = pdf_filepath
    else:
        print("pdf_filepath: {}".format(pdf_filepath))
        new_pdf_filepath = os.path.join(
            pdf_folderpath, "comic_book_{}.pdf".format(random.randint(0, 10))
        )
        print("new pdf_filepath: {}".format(new_pdf_filepath))

    # Open the PDF file in binary mode
    with open(new_pdf_filepath, "rb") as file:
        # Create a BytesIO object to hold the streamed data
        file_like = BytesIO(file.read())
        # Return the streamed response
        return StreamingResponse(file_like, media_type="application/pdf")


@app.post("/compare_books", status_code=200)
async def get_explanation_comparision(selected_book_for_comparison_lst: dict):
    print(selected_book_for_comparison_lst)
    comparision_dict = {"compared_books": []}
    print(selected_book_for_comparison_lst)
    for obj in selected_book_for_comparison_lst["selected_book_lst"]:
        tmp_dict = {}
        selected_book_id = obj["comic_no"]
        tmp_dict = {
            selected_book_id: {
                "story_pace": {},
                "characters": {},
                "genres": {},
                "book_cover": {},
            }
        }
        tmp_dict[selected_book_id]["story_pace"] = le_utils.find_numpages_and_storypace(
            selected_book_id
        )
        tmp_dict[selected_book_id][
            "characters"
        ] = le_utils.fetch_character_info_for_local_explanation(selected_book_id)
        tmp_dict[selected_book_id]["genres"] = fetch_genres(selected_book_id)
        tmp_dict[selected_book_id]["book_cover"] = le_utils.fetch_book_cover_keywords(
            selected_book_id
        )

        comparision_dict["compared_books"].append(tmp_dict)

    # log data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "selected_book_for_comparison_lst": selected_book_for_comparison_lst
            },
            "output_data": comparision_dict,
            "function_name": "get_explanation_comparision",
        },
    )

    return comparision_dict


@app.post("/book_search_with_no_personalization", status_code=200)
async def search_with_no_reranking(
    cbl: BookList,
    b_id: int = Query(...),
    generate_fake_clicks: bool = Query(default=True),
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        cld=1.0,
        edh=1.0,
        hog=1.0,
        text=1.0,
        comic_img=1.0,
        comic_txt=1.0,
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=0.01,
        comic_cover_txt=0.01,
    ),
):
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_coarse_search_without_reranking(
        b_id=b_id, feature_weight_dict=input_feature_importance_dict.dict(), top_n=19
    )

    print(coarse_filtered_book_df.head(100))
    print(coarse_filtered_book_new_lst)
    print(
        "generate_fake_clicks: {} | {}".format(
            generate_fake_clicks, type(generate_fake_clicks)
        )
    )

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )

    else:
        clicksinfo_dict = create_real_clicks_for_previous_timestamp_data(
            cbl.interested_book_lst
        )  # [{"comic_no": "1", "clicked": 1.0}, {"comic_no": "3", "clicked": 0.0}]
    print("clicksinfo_dict: {}".format(clicksinfo_dict))

    normalized_feature_importance_dict = {
        "gender": round(random.random(), 2),
        "supersense": round(random.random(), 2),
        "genre_comb": round(random.random(), 2),
        "panel_ratio": round(random.random(), 2),
        "comic_cover_img": round(random.random(), 2),
        "comic_cover_txt": round(random.random(), 2),
    }
    clf_coef = None
    feature_importance_dict = normalized_feature_importance_dict
    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )

    if generate_fake_clicks:
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=[],
            query_book_id=b_id,
            search_results=coarse_filtered_book_new_lst,
            model=sentence_transformer_model,
        )
    else:

        print("clicksinfo_dict: {}".format(clicksinfo_dict))
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=clicksinfo_dict,
            query_book_id=b_id,
            search_results=coarse_filtered_book_new_lst,
            model=sentence_transformer_model,
        )

    # add facet weights to UI dict
    print(
        {
            **coarse_filtered_book_new_lst[0],
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
    )
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(coarse_filtered_book_new_lst)
        if idx <= 20
    ]

    print(feature_importance_dict, clf_coef)
    interpretable_filtered_book_new_lst = [
        coarse_filtered_book_new_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    print()
    print(" +++++++++++++ ++++++++++++ +++++++++++++ ")
    print()

    for x in coarse_filtered_book_new_lst:
        print(x)

    print()
    print(" +++++++++++++ ++++++++++++ ++++++++++++++ ")
    print()

    # log userrs interaction data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "cbl": cbl.dict(),
                "b_id": b_id,
                "generate_fake_clicks": generate_fake_clicks,
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "search_with_no_reranking",
        },
    )
    return interpretable_filtered_book_new_lst


@app.post("/book_search_with_searchbar_inputs_and_random_serp", status_code=200)
async def search_with_searchbar_inputs_with_random_serp(
    searchbar_query: SearchBarQuery,
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=1.0,
        comic_cover_txt=1.0,
    ),
):
    print("searchbar_query : {}".format(searchbar_query))

    # if you use search results from search bar reset history
    global book_search_results_history_lst
    book_search_results_history_lst = []

    if (
        searchbar_query.type == "book"
        or searchbar_query.type == "character"
        or searchbar_query.type == "genre"
    ):
        b_id = searchbar_query.comic_no
        (
            coarse_filtered_book_new_lst,
            coarse_filtered_book_df,
        ) = cs_utils.perform_random_search(
            b_id=b_id,
            feature_weight_dict=input_feature_importance_dict.dict(),
            top_n=19,
        )
    else:
        b_id = 1
        (
            coarse_filtered_book_new_lst,
            coarse_filtered_book_df,
        ) = cs_utils.perform_random_search(
            b_id=b_id,
            feature_weight_dict=input_feature_importance_dict.dict(),
            top_n=19,
        )

    normalized_feature_importance_dict = {
        "gender": input_feature_importance_dict.gender,
        "supersense": input_feature_importance_dict.supersense,
        "genre_comb": input_feature_importance_dict.genre_comb,
        "panel_ratio": input_feature_importance_dict.panel_ratio,
        "comic_cover_img": input_feature_importance_dict.comic_cover_img,
        "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
    }

    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )
    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
        historical_book_ids_lst,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
        historical_book_ids_lst=book_search_results_history_lst.copy(),
    )
    book_search_results_history_lst = historical_book_ids_lst.copy()
    print("interpretable_filtered_book_lst: {}".format(
        interpretable_filtered_book_lst))
    relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
        clicksinfo_dict=[],
        query_book_id=b_id,
        search_results=interpretable_filtered_book_lst,
        model=sentence_transformer_model,
    )

    print("relevance_feedback_explanation_dict: {}".format(
        relevance_feedback_explanation_dict))

    # add facet weights to UI
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(interpretable_filtered_book_lst)
        if idx <= 20
    ]
    print(
        {**interpretable_filtered_book_lst[0], **normalized_feature_importance_dict})
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    # log data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "searchbar_query": searchbar_query.dict(),
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "search_with_searchbar_inputs_with_random_serp",
        },
    )

    return interpretable_filtered_book_new_lst


@app.post("/book_search_with_random_serp_results", status_code=200)
async def search_with_real_clicks_and_random_serp(
    cbl: BookList,
    b_id: int = Query(...),
    generate_fake_clicks: bool = Query(default=True),
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=1.0,
        comic_cover_txt=1.0,
    ),
):
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_random_search(
        b_id=b_id, feature_weight_dict=input_feature_importance_dict.dict(), top_n=19
    )

    print(coarse_filtered_book_df.head(100))
    print(coarse_filtered_book_new_lst)
    print(
        "generate_fake_clicks: {} | {}".format(
            generate_fake_clicks, type(generate_fake_clicks)
        )
    )

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )

    else:
        clicksinfo_dict = create_real_clicks_for_previous_timestamp_data(
            cbl.interested_book_lst
        )  # [{"comic_no": "1", "clicked": 1.0}, {"comic_no": "3", "clicked": 0.0}]
    print("clicksinfo_dict: {}".format(clicksinfo_dict))

    normalized_feature_importance_dict = {
        "gender": round(random.random(), 2),
        "supersense": round(random.random(), 2),
        "genre_comb": round(random.random(), 2),
        "panel_ratio": round(random.random(), 2),
        "comic_cover_img": round(random.random(), 2),
        "comic_cover_txt": round(random.random(), 2),
    }
    clf_coef = None
    feature_importance_dict = normalized_feature_importance_dict
    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )

    if generate_fake_clicks:
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=[],
            query_book_id=b_id,
            search_results=coarse_filtered_book_new_lst,
            model=sentence_transformer_model,
        )
    else:

        print("clicksinfo_dict: {}".format(clicksinfo_dict))
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback(
            clicksinfo_dict=clicksinfo_dict,
            query_book_id=b_id,
            search_results=coarse_filtered_book_new_lst,
            model=sentence_transformer_model,
        )

    # add facet weights to UI dict
    print(
        {
            **coarse_filtered_book_new_lst[0],
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
    )
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(coarse_filtered_book_new_lst)
        if idx <= 20
    ]

    print(feature_importance_dict, clf_coef)
    interpretable_filtered_book_new_lst = [
        coarse_filtered_book_new_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    print()
    print(" +++++++++++++ ++++++++++++ +++++++++++++ ")
    print()

    for x in coarse_filtered_book_new_lst:
        print(x)

    print()
    print(" +++++++++++++ ++++++++++++ ++++++++++++++ ")
    print()

    # log userrs interaction data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "cbl": cbl.dict(),
                "b_id": b_id,
                "generate_fake_clicks": generate_fake_clicks,
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "search_with_real_clicks_and_random_serp",
        },
    )
    return interpretable_filtered_book_new_lst


@app.post("/book_search_with_random_explanation_feedback", status_code=200)
async def search_with_real_clicks_and_random_explanation_feedback(
    cbl: BookList,
    b_id: int = Query(...),
    generate_fake_clicks: bool = Query(default=True),
    input_feature_importance_dict: Optional[FacetWeight] = FacetWeight(
        gender=1.0,
        supersense=1.0,
        genre_comb=1.0,
        panel_ratio=1.0,
        comic_cover_img=1.0,
        comic_cover_txt=1.0,
    ),
):
    # print(cbl.clicked_book_lst)
    # print(b_id, generate_fake_clicks)
    # b_id: int, clicksinfo_dict: dict, generate_fake_clicks=True
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_coarse_search(b_id=b_id)

    print(coarse_filtered_book_df.head(100))
    print(
        "generate_fake_clicks: {} | {}".format(
            generate_fake_clicks, type(generate_fake_clicks)
        )
    )

    # if you use search results from search bar reset history
    global book_search_results_history_lst

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )

    else:
        clicksinfo_dict = create_real_clicks_for_previous_timestamp_data(
            cbl.interested_book_lst
        )  # [{"comic_no": "1", "clicked": 1.0}, {"comic_no": "3", "clicked": 0.0}]
    print("clicksinfo_dict: {}".format(clicksinfo_dict))

    if not generate_fake_clicks:

        # handle if user directly clikcs on book without hovering and if all books are hovered
        if utils.check_if_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ) and not utils.check_if_all_books_are_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ):
            (
                feature_importance_dict,
                normalized_feature_importance_dict,
                clf_coef,
            ) = rrr.adapt_facet_weights_from_previous_timestep_click_info_triplet_loss(
                previous_click_info_lst=clicksinfo_dict, query_book_id=b_id
            )
        elif not utils.check_if_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ) and utils.check_if_all_books_are_hovered(
            clicksinfo_dict=clicksinfo_dict, b_id=b_id
        ):
            normalized_feature_importance_dict = {
                "gender": input_feature_importance_dict.gender,
                "supersense": input_feature_importance_dict.supersense,
                "genre_comb": input_feature_importance_dict.genre_comb,
                "panel_ratio": input_feature_importance_dict.panel_ratio,
                "comic_cover_img": input_feature_importance_dict.comic_cover_img,
                "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
            }
            clf_coef = None
            feature_importance_dict = normalized_feature_importance_dict
        else:
            normalized_feature_importance_dict = {
                "gender": input_feature_importance_dict.gender,
                "supersense": input_feature_importance_dict.supersense,
                "genre_comb": input_feature_importance_dict.genre_comb,
                "panel_ratio": input_feature_importance_dict.panel_ratio,
                "comic_cover_img": input_feature_importance_dict.comic_cover_img,
                "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
            }
        clf_coef = None
        feature_importance_dict = normalized_feature_importance_dict
    else:
        normalized_feature_importance_dict = {
            "gender": input_feature_importance_dict.gender,
            "supersense": input_feature_importance_dict.supersense,
            "genre_comb": input_feature_importance_dict.genre_comb,
            "panel_ratio": input_feature_importance_dict.panel_ratio,
            "comic_cover_img": input_feature_importance_dict.comic_cover_img,
            "comic_cover_txt": input_feature_importance_dict.comic_cover_txt,
        }
        clf_coef = None
        feature_importance_dict = normalized_feature_importance_dict

    print(
        "normalized_feature_importance_dict: {}".format(
            normalized_feature_importance_dict
        )
    )

    # new_normalized_feature_importance_dict = {"gender": 1.0, "supersense": 1.0, "genre_comb": 1.0, "panel_ratio": 1.0, "comic_cover_img": 1.0, "comic_cover_txt": 1.0}
    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
        historical_book_ids_lst,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
        historical_book_ids_lst=book_search_results_history_lst.copy(),
    )

    book_search_results_history_lst = historical_book_ids_lst.copy()

    if generate_fake_clicks:
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback_at_random(
            clicksinfo_dict=[],
            query_book_id=b_id,
            search_results=interpretable_filtered_book_lst,
            model=sentence_transformer_model,
        )
    else:

        print("clicksinfo_dict: {}".format(clicksinfo_dict))
        relevance_feedback_explanation_dict = await erf.explain_relevance_feedback_at_random(
            clicksinfo_dict=clicksinfo_dict,
            query_book_id=b_id,
            search_results=interpretable_filtered_book_lst,
            model=sentence_transformer_model,
        )

    # add facet weights to UI dict
    print(
        {
            **interpretable_filtered_book_lst[0],
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
    )
    interpretable_filtered_book_new_lst = [
        {
            **d,
            **normalized_feature_importance_dict,
            **relevance_feedback_explanation_dict,
        }
        for idx, d in enumerate(interpretable_filtered_book_lst)
        if idx <= 20
    ]

    print(feature_importance_dict, clf_coef)
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
        relevance_feedback_explanation_dict,
    ]

    print()
    print(" +++++++++++++ ++++++++++++ +++++++++++++ ")
    print()

    for x in interpretable_filtered_book_lst:
        print(x)

    print()
    print(" +++++++++++++ ++++++++++++ ++++++++++++++ ")
    print()

    # log userrs interaction data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "cbl": cbl.dict(),
                "b_id": b_id,
                "generate_fake_clicks": generate_fake_clicks,
                "input_feature_importance_dict": input_feature_importance_dict.dict(),
            },
            "output_data": {
                "interpretable_filtered_book_new_lst": interpretable_filtered_book_new_lst
            },
            "function_name": "book_search_with_random_explanation_feedback",
        },
    )
    return interpretable_filtered_book_new_lst


@app.post("/compare_books_with_random", status_code=200)
async def get_explanation_comparision_with_random(
    selected_book_for_comparison_lst: dict,
):

    print(selected_book_for_comparison_lst)
    comparision_dict = {"compared_books": []}
    selected_book_ids_for_comparison_lst = [
        obj["comic_no"] for obj in selected_book_for_comparison_lst["selected_book_lst"]
    ]
    all_books_ids_lst = [
        x for x in range(1, 1712) if x not in selected_book_ids_for_comparison_lst
    ]
    random_book_ids_lst = np.random.choice(
        all_books_ids_lst, size=len(selected_book_ids_for_comparison_lst), replace=False
    ).tolist()
    print(
        random_book_ids_lst,
        selected_book_ids_for_comparison_lst,
        len(selected_book_for_comparison_lst),
    )
    for idx, obj in enumerate(selected_book_for_comparison_lst["selected_book_lst"]):
        tmp_dict = {}
        selected_book_id = obj["comic_no"]
        tmp_dict = {
            selected_book_id: {
                "story_pace": {},
                "characters": {},
                "genres": {},
                "book_cover": {},
            }
        }
        tmp_dict[selected_book_id]["story_pace"] = le_utils.find_numpages_and_storypace(
            random_book_ids_lst[idx]
        )
        tmp_dict[selected_book_id][
            "characters"
        ] = le_utils.fetch_character_info_for_local_explanation(
            random_book_ids_lst[idx]
        )
        tmp_dict[selected_book_id]["genres"] = fetch_genres(
            random_book_ids_lst[idx])
        tmp_dict[selected_book_id]["book_cover"] = le_utils.fetch_book_cover_keywords(
            random_book_ids_lst[idx]
        )
        comparision_dict["compared_books"].append(tmp_dict)

    # log data for evaluation
    global latest_session_folderpath
    utils.log_session_data(
        latest_session_folderpath,
        {
            "input_data": {
                "selected_book_for_comparison_lst": selected_book_for_comparison_lst
            },
            "output_data": comparision_dict,
            "function_name": "get_explanation_comparision_with_random",
        },
    )

    return comparision_dict


@app.get("/rooturl")
async def read_root():
    return {"message": "Hello from FastAPI"}


@app.get("/test")
def read_main(request: Request):
    return {"message": "Hello World", "root_path": request.scope.get("root_path")}


if __name__ == "__main__":
    # get entry page results
    coarse_filtered_book_new_lst, coarse_filtered_book_df = get_coarse_results(
        542)
    clicked_book_lst = create_fake_clicks_for_previous_timestep_data(
        coarse_filtered_book_df
    )
    print(clicked_book_lst[:20])
