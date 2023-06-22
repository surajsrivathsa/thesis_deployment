import os, sys, glob, pathlib, re
import pandas as pd, numpy as np
import time
import pickle
import time
from multiprocessing import Pool as ProcessPool

sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis\thesis_comics_search_xai\jupyter_notebooks")
sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis\thesis_comics_search_xai")
sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis")

# custom imports
import common_functions.backend_utils as utils
import common_functions.img_to_text_extractor as img_txt_extract
  
    
if __name__ == '__main__':
    
    parent_folderpath = r"C:\Users\Suraj Shashidhar\Documents\thesis\famous_comics_titles"
    comic_title_folderpath = ["archie", "asterix",  "avengers", "captain_america", "captain_marvel", "captain_steven_savage","conan_the_barbarian", "daredevil",
                            "donald_duck", "fantastic_four", "flash", "flintstones", "ghost_rider", "hulk", "iron_man", "jetsons", "lucky_luke", "nick_fury", "rawhide_kid",
                            "sandman", "silver_surfer", "spiderman", "superman", "tarzan", "the_boys", "thor", "tin-tin", "watchmen", "x-men" ]


    book_metadata_dict, comic_book_metadata_df =  utils.load_book_metadata()
    
    
    # individual_comic_lst = comic_book_metadata_df.loc[ comic_book_metadata_df['our_idx'] >= 499 , 'Book Title'].tolist()
    # parent_folderpath_lst = comic_book_metadata_df.loc[ comic_book_metadata_df['our_idx'] >= 499 , 'img_folderpath'].tolist()
    # print(len(individual_comic_lst))
    
    full_records_dict_lst = []
    required_issues_lst = ['asterix','funny_man', 'conan_the_barbarian']
    # ['archie', 'asterix', 'captain_marvel', 'conan_the_barbarian', 'donald_duck', 'fantastic_four', 'flintstones', 'funny_man', 'ghost_rider', 'green_lantern', 'jetsons', 'power_rangers', 'tarzan', 'the_boys', 'tin-tin', 'watchmen' ]
    #["aquaman", "batman", "blondie", "from_hell", "funny_man", "justice_league", "justice_society", "maus", "nick_fury", "persepolis", "power_rangers", "spectre" ,"tomb_raider", "wonder_woman" ]
    filtered_comic_book_metadata_df = comic_book_metadata_df[comic_book_metadata_df['Issue'].isin(required_issues_lst)]
    
    individual_comic_lst = filtered_comic_book_metadata_df['Book Title'].tolist()
    parent_folderpath_lst = filtered_comic_book_metadata_df['img_folderpath'].tolist()
    our_idx_lst = filtered_comic_book_metadata_df['our_idx'].tolist()
    comic_no_lst = filtered_comic_book_metadata_df['comic_no'].tolist()
    text_folderpath_lst = [os.path.join(os.path.dirname(di), 'text') for di in parent_folderpath_lst]
    our_idx = 499
    comic_no = 3950

    for idx, (ind_cmc, parent_fldr, our_idx, comic_no, text_fldr) in enumerate(zip( individual_comic_lst, parent_folderpath_lst, our_idx_lst, comic_no_lst, text_folderpath_lst )):
        record = {'comic_id': comic_no, 'individual_comic_name': ind_cmc, 'parent_folder_path': parent_fldr, 'our_idx': our_idx , 'text_folder_path': text_fldr}
        full_records_dict_lst.append(record)
    
    print('books to be processed: {}'.format(len(full_records_dict_lst)))
    print()
    print(full_records_dict_lst[:3])
    print()
    print(full_records_dict_lst[-3:])
    print()
    
    
    start_time = time.time()
    
    with ProcessPool(4) as prcs_pool:
        # execute tasks in chunks, block until all complete
        all_book_record_lst = prcs_pool.map(img_txt_extract.process_comic_book_text, full_records_dict_lst, chunksize=2)
    
    end_time = time.time()
    
    a = {'all_book_record_lst': all_book_record_lst}
    print()
    
    print('processing time: {}'.format((end_time-start_time)/60.0))
    
    
    with open('all_book_record_lst.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    flat_list = [item for sublist in all_book_record_lst for item in sublist]
    all_text_df = pd.DataFrame.from_records(flat_list)
    all_text_df.to_csv('all_text_df.csv', index=False)
    print()  
    print('pages processed: {}'.format(len(flat_list)))
    print()
    print(all_text_df.head()) # 10789 pages/ 170 min
    