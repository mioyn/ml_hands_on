import json
from pathlib import Path

jsonl_path = Path("datasets/meta_Electronics/meta_Electronics.jsonl")
unique = set()

# read jsonl file line by line
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        main_category = data.get("main_category")
        if main_category:
            unique.add(main_category)


print("unique main_category count:", len(unique))

# write list to file
# with open(
#     "datasets/meta_Electronics/unique_main_category.txt", "w", encoding="utf-8"
# ) as out:
#     for a in sorted(unique):
#         out.write(a + "\n")


# create csv files for each main_category with fields as below:
# Field      Type      Explanation
# main_category      str      Main category (i.e., domain) of the product.
# title      str      Name of the product.
# average_rating      float      Rating of the product shown on the product page.
# rating_number      int      Number of ratings in the product.
# features      list      Bullet-point format features of the product.
# description      list      Description of the product.
# price      float      Price in US dollars (at time of crawling).
# images      list      Images of the product. Each image has different sizes (thumb, large, hi_res). The “variant” field shows the position of image.
# videos      list      Videos of the product including title and url.
# store      str      Store name of the product.
# categories      list      Hierarchical categories of the product.
# details      dict      Product details, including materials, brand, sizes, etc.
# parent_asin      str      Parent ID of the product.
# bought_together     list      Recommended bundles from the websites.

for category in unique:
    category_path = Path(f"datasets/meta_Electronics/main_category/{category}.csv")
    category_path.parent.mkdir(parents=True, exist_ok=True)
    with category_path.open("w", encoding="utf-8") as out:
        # write header
        out.write(
            "main_category,title,average_rating,rating_number,price,store,parent_asin\n"
        )
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if (
                    data.get("main_category") == category
                    and data.get("rating_number") >= 500
                ):
                    # prepare csv line
                    csv_line = (
                        f'{data.get("main_category","")},'
                        f'{data.get("title","").replace(",","| ") if data.get("title") else ""},'
                        f'{data.get("average_rating","") if data.get("average_rating") else ""},'
                        f'{data.get("rating_number","") if data.get("rating_number") else ""},'
                        f'{data.get("price","") if data.get("price") else ""},'
                        f'{data.get("store","").replace(",","| ") if data.get("store") else ""},'
                        f'{data.get("parent_asin","")}\n'
                    )
                    out.write(csv_line)


# for category in unique:
#     category_path = Path(f"datasets/meta_Electronics/main_category/{category}.jsonl")
#     category_path.parent.mkdir(parents=True, exist_ok=True)
#     with category_path.open("w", encoding="utf-8") as out:
#         with jsonl_path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 if data.get("main_category") == category:
#                     out.write(line)
print("done")
