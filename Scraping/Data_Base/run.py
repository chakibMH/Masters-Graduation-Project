# folder = "C:/Users/chaki/Desktop/M2 SII/_PFE/PFE_code/LOCAL_RESULTS_NO_GITHUB/ACM_Scraping/merge"

# def setup():
#     db = merge_data_sets(folder)
#     return db


# db = setup()

db = pd.read_csv("final_data_non_clean.csv")


# next:
clean_batch = clean_DB(db.iloc[130000:])




