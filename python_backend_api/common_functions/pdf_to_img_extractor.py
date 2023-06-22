import os, sys, glob
import fitz
from PIL import Image
import time
import traceback
import multiprocessing

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

def convert_pdf_page_to_img(page, doc_folderpath, mat):
    try:
        # pix = page.get_pixmap(matrix=mat)  # render page to an image
        pix = page.get_pixmap()
        h = pix.height
        w = pix.width
        pg_number = page.number
        img_filename = str(pg_number).zfill(5) + '.jpeg'
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        if h < 600 or w < 600:
            h = pix.height
            w = pix.width
            multiplication_factor = max(int( 1200/(1e-2 + h) ), int( 1200/(1e-2 + w) ))
            pix = page.get_pixmap(matrix=fitz.Matrix( min(max(zoom_x, multiplication_factor), 8) , min(max(zoom_y, multiplication_factor), 8) ))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # img = img.resize(( int(pix.width * multiplication_factor), int(pix.height * multiplication_factor) ), Image.ANTIALIAS)
            
        elif h > 1600 or w > 1600:
            img = img.resize(( int(w * 0.5) , int(h * 0.5) ), Image.ANTIALIAS)
        
        img.save(os.path.join(doc_folderpath, img_filename) ,quality=20, optimize=True)
        
        # failure_msg = ''
        # run_status = True
        # pg_filepath = os.path.join(doc_folderpath, img_filename)
    except Exception as e:
        print('Cannot process page: {}'.format(page), flush=True)
        print(traceback.format_exc())
        # pg_number = page.number
        # img_filename = 'pg_' + str(pg_number).zfill(5) + '.jpeg'
        # failure_msg = traceback.format_exc()
        # run_status = False
        # pg_filepath = os.path.join(doc_folderpath, img_filename)
    finally:
         
        return 
    
    
    
def process_one_comic_title(pdf_folderpath: str, page_img_folderpath: str):
    # To get better resolution
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    
    time_taken_to_process_doc_lst = []
    print('', flush=True)
    curr_proc = multiprocessing.current_process()
    print('Folder {} processed by : {}'.format(pdf_folderpath, curr_proc.name), flush=True)
    for pdf_file in glob.glob(  os.path.join(pdf_folderpath, "*.pdf")  ):
        start_time = time.time()
        print('Processing: {}'.format(pdf_file), flush=True)
        print('', flush=True)
        pdf_doc = fitz.open(pdf_file)  # open document
        pdf_file_basename = os.path.basename(pdf_file).replace('.pdf', '')
        
        # if the folder doesnt exist then create
        if not os.path.exists( os.path.join(page_img_folderpath, pdf_file_basename) ):
            os.mkdir(os.path.join(page_img_folderpath, pdf_file_basename) )
            
        doc_folderpath = os.path.join(page_img_folderpath, pdf_file_basename) 
        
        # num_threads = 4
        # pool = ThreadPool(num_threads)

        # run_results = pool.starmap( convert_pdf_page_to_img, zip( [ x for x in pdf_doc.pages()], [ doc_folderpath for x in pdf_doc.pages()], [ mat for x in pdf_doc.pages()] ) )
        
        # # Close the pool and wait for the work to finish
        # pool.close()
        # pool.join()
        # print()
        doc_pages_lst = [ x for x in pdf_doc.pages()]; folderpath_lst = [ doc_folderpath for x in pdf_doc.pages()]; matrix_lst =  [ mat for x in pdf_doc.pages()];
        for idx, (page, doc_folderpath, mat) in enumerate(zip(doc_pages_lst, folderpath_lst, matrix_lst )):
            
            convert_pdf_page_to_img(page, doc_folderpath, mat)
        
        
        end_time = time.time()
        time_taken_to_process_doc_lst.append( (end_time-start_time)/60.0 )
    
    return time_taken_to_process_doc_lst