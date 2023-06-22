import os, sys
import statistics
from multiprocessing import Pool as ProcessPool

sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis\thesis_comics_search_xai\jupyter_notebooks")
sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis\thesis_comics_search_xai")
sys.path.insert(0, r"C:\Users\Suraj Shashidhar\Documents\thesis")

# custom imports
import common_functions.backend_utils as utils
import common_functions.pdf_to_img_extractor as pdf_img_extract


   
if __name__ == '__main__':
    
    parent_folderpath = r"C:\Users\Suraj Shashidhar\Documents\thesis\famous_comics_titles"
    comic_title_folderpath = [ "justice_society", "power_rangers", "spectre",] 
    
    time_taken_to_process_doc_lst = []
    comic_parent_folderpath_lst = []
    page_img_folderpath_lst = []
    text_folderpath_lst = []
    pdf_folderpath_lst = []


    for ctf in comic_title_folderpath:
        comic_parent_folderpath = os.path.join(parent_folderpath, ctf)
        
        pdf_folderpath = os.path.join(comic_parent_folderpath, "pdf")
        page_img_folderpath = os.path.join(comic_parent_folderpath, "page_img")
        text_folderpath = os.path.join(comic_parent_folderpath, "text")
        
        comic_parent_folderpath_lst.append(comic_parent_folderpath)
        pdf_folderpath_lst.append(pdf_folderpath)
        page_img_folderpath_lst.append(page_img_folderpath)
        text_folderpath_lst.append(text_folderpath)

    print(len(comic_parent_folderpath_lst), len(pdf_folderpath_lst), len(page_img_folderpath_lst), len(text_folderpath_lst) )

    num_prcs = 3
    prcs_pool = ProcessPool(num_prcs)
    time_taken_to_process_doc_lst = prcs_pool.starmap(pdf_img_extract.process_one_comic_title, zip(pdf_folderpath_lst, page_img_folderpath_lst), chunksize=1)
    prcs_pool.close()
    prcs_pool.join()
    
    # print('average time taken per pdf: {}'.format(statistics.mean(time_taken_to_process_doc_lst)))
    # print('total time taken: {}'.format(sum(time_taken_to_process_doc_lst)))
    