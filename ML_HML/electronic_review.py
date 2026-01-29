import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import json
    from pathlib import Path
    import csv
    import pandas as pd
    return Path, csv, json, pd


@app.cell
def _():
    SMART_HOME_KEYWORDS = [
        "smart", "wifi", "wi-fi", "alexa", "echo",
        "google home", "nest", "zigbee", "z-wave",
        "iot", "connected", "matter", "home automation"
    ]
    EXCLUDE_KEYWORDS = [
        "case", "cover", "mount", "holder", "stand",
        "adapter", "cable", "charger", "replacement",
        "battery", "accessory"
    ]

    return EXCLUDE_KEYWORDS, SMART_HOME_KEYWORDS


@app.cell
def _():
    SMART_HOME_KEYWORDS = [
        "smart",
        "wifi",
        "wi-fi",
        "wireless",
        "zigbee",
        "z-wave",
        "bluetooth",
        "iot",
        "internet of things"
    ]
    SMART_HOME_KEYWORDS += [
        "smart lock",
        "video doorbell",
        "doorbell camera",
        "security camera",
        "smart camera",
        "motion sensor",
        "contact sensor",
        "leak sensor",
        "smoke detector",
        "carbon monoxide",
        "alarm system"
    ]
    SMART_HOME_KEYWORDS += [
        "smart bulb",
        "smart light",
        "smart switch",
        "smart plug",
        "smart outlet",
        "dimmer",
        "led strip",
        "light strip",
        "power strip"
    ]
    SMART_HOME_KEYWORDS += [
        "smart thermostat",
        "thermostat",
        "temperature sensor",
        "humidity sensor",
        "energy monitor",
        "smart meter"
    ]
    SMART_HOME_KEYWORDS += [
        "robot vacuum",
        "robotic vacuum",
        "smart vacuum",
        "air purifier",
        "smart fan",
        "smart heater",
        "smart humidifier",
        "dehumidifier"
    ]
    SMART_HOME_KEYWORDS += [
        "smart hub",
        "home hub",
        "gateway",
        "bridge",
        "controller"
    ]

    EXCLUDE_KEYWORDS = [
        "case",
        "cover",
        "skin",
        "protector",
        "screen protector",
        "mount",
        "stand",
        "holder",
        "clip",
        "strap",
        "band",
        "replacement",
        "refill"
    ]
    EXCLUDE_KEYWORDS += [
        "cable",
        "adapter",
        "charger",
        "battery",
        "power supply",
        "usb",
        "hdmi",
        "ethernet",
        "switch cable"
    ]
    EXCLUDE_KEYWORDS += [
        "headphones",
        "earbuds",
        "speaker",
        "smartwatch",
        "fitness tracker",
        "vr",
        "gaming",
        "controller",
        "console"
    ]
    EXCLUDE_KEYWORDS += [
        "laptop",
        "notebook",
        "tablet",
        "phone",
        "smartphone",
        "iphone",
        "android",
        "monitor",
        "keyboard",
        "mouse"
    ]

    EXCLUDE_KEYWORDS += [
        "bag",
        "backpack",
        "wallet",
        "watch band",
        "bracelet",
        "jewelry",
        "sticker",
        "decal"
    ]

    return EXCLUDE_KEYWORDS, SMART_HOME_KEYWORDS


@app.cell
def _(EXCLUDE_KEYWORDS, Path, SMART_HOME_KEYWORDS, csv, json):

    def _():

        out_path = Path("datasets/filtered_smart_home_reviews/filtered_smart_home_reviews2.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)  # make dirs if needed

        def contains_any(text, keywords):
            text = text.lower()
            return any(k in text for k in keywords)

        with open("datasets/filtered_smart_home_reviews/filtered_smart_home_reviews2.csv", "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow([
                "asin", "parent_asin", "timestamp",
                "rating", "title", "text"
            ])

            with open("datasets/ElectronicsRe/Electronics.jsonl", "r") as f:
                for line in f:
                    obj = json.loads(line)

                    title = (obj.get("title") or "").lower()
                    text = (obj.get("text") or "").lower()
                    full_text = title + " " + text

                    if not contains_any(full_text, SMART_HOME_KEYWORDS):
                        continue
                    if contains_any(full_text, EXCLUDE_KEYWORDS):
                        continue
                    writer.writerow([
                        obj["asin"],
                        obj.get("parent_asin"),
                        obj["timestamp"],
                        obj["rating"],
                        obj.get("title"),
                        obj.get("text")
                    ])

        print("Done filtering smart home reviews.")
    _()
    return


@app.cell
def _(EXCLUDE_KEYWORDS, SMART_HOME_KEYWORDS, csv, json):

    def _():
        def contains_any(text, keywords):
            text = text.lower()
            return any(k in text for k in keywords)

        with open("datasets/filtered_smart_home_reviews/filtered_smart_home_reviews2.csv", "a", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)

            with open("datasets/Home_and_Kitchen/Home_and_Kitchen.jsonl", "r") as f:
                for line in f:
                    obj = json.loads(line)

                    title = (obj.get("title") or "").lower()
                    text = (obj.get("text") or "").lower()
                    full_text = title + " " + text

                    if not contains_any(full_text, SMART_HOME_KEYWORDS):
                        continue
                    if contains_any(full_text, EXCLUDE_KEYWORDS):
                        continue

                    writer.writerow([
                        obj["asin"],
                        obj.get("parent_asin"),
                        obj["timestamp"],
                        obj["rating"],
                        obj.get("title"),
                        obj.get("text")
                    ])


    _()
    return


@app.cell
def _(pd):

    chunk_size = 200_000

    asin_counts = {}
    asin_min_ts = {}
    asin_max_ts = {}

    for chunk in pd.read_csv(
        "datasets/filtered_smart_home_reviews/filtered_smart_home_reviews2.csv",
        chunksize=chunk_size
    ):
        for asin, g in chunk.groupby("asin"):
            asin_counts[asin] = asin_counts.get(asin, 0) + len(g)
            asin_min_ts[asin] = min(
                asin_min_ts.get(asin, g["timestamp"].min()),
                g["timestamp"].min()
            )
            asin_max_ts[asin] = max(
                asin_max_ts.get(asin, g["timestamp"].max()),
                g["timestamp"].max()
            )
    return asin_counts, asin_max_ts, asin_min_ts, chunk_size


@app.cell
def _(asin_counts, asin_max_ts, asin_min_ts, pd):

    asin_df = pd.DataFrame({
        "asin": asin_counts.keys(),
        "review_count": asin_counts.values(),
        "first_ts": [asin_min_ts[a] for a in asin_counts],
        "last_ts": [asin_max_ts[a] for a in asin_counts],
    })

    asin_df["span_days"] = (
        pd.to_datetime(asin_df["last_ts"], unit="ms") -
        pd.to_datetime(asin_df["first_ts"], unit="ms")
    ).dt.days

    asin_df = asin_df[
        (asin_df["review_count"] >= 800) &     # strong signal
        (asin_df["span_days"] >= 730)           # ≥ 8 quarters
    ]
    return (asin_df,)


@app.cell
def _(asin_df):
    asin_ranked = asin_df.sort_values("review_count", ascending=False)

    top_10_asins = asin_ranked.head(10)["asin"].tolist()
    top_20_asins = asin_ranked.head(20)["asin"].tolist()
    top_15_to_30_asins = asin_ranked.iloc[15:30]["asin"].tolist()
    print("Top 10 ASINs  :", top_10_asins)
    print("Top 20 ASINs  :", top_20_asins)
    print("ASINs 15 to 30:", top_15_to_30_asins)
    return top_10_asins, top_15_to_30_asins, top_20_asins


@app.cell
def _(chunk_size, pd):
    def save_asins_to_file(asins, filename):
        final_rows = []

        for chunk in pd.read_csv(
            "datasets/filtered_smart_home_reviews/filtered_smart_home_reviews2.csv",
            chunksize=chunk_size
        ):
            chunk = chunk[chunk["asin"].isin(asins)]
            final_rows.append(chunk)

        final_df = pd.concat(final_rows)
        final_df.to_csv(filename, index=False)
        print(f"Saved ASIN reviews to {filename}.")
    return (save_asins_to_file,)


@app.cell
def _(save_asins_to_file, top_10_asins, top_15_to_30_asins, top_20_asins):

    save_asins_to_file(top_10_asins, "datasets/filtered_smart_home_reviews/final_top_10_smart_home_products1.csv")
    save_asins_to_file(top_20_asins, "datasets/filtered_smart_home_reviews/final_top_20_smart_home_products1.csv")
    save_asins_to_file(top_15_to_30_asins, "datasets/filtered_smart_home_reviews/final_asins_15_to_30_smart_home_products1.csv")
    return


@app.cell
def _(csv, json):

    def extract_from_meta(
        meta_path,
        output_csv,
        target_asins,
        mode="w"
    ):
        target_asins = set(target_asins)

        fields = [
            "parent_asin",
            "title",
            "store",
            "main_category",
            "categories",
            "average_rating",
            "rating_number",
            "price",
            "date_first_available"
        ]

        with open(output_csv, mode, newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            if mode == "w":
                writer.writeheader()

            with open(meta_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    asin = obj.get("parent_asin")
                    if asin not in target_asins:
                        continue

                    details = obj.get("details", {}) or {}

                    row = {
                        "parent_asin": asin,
                        "title": obj.get("title"),
                        "store": obj.get("store"),
                        "main_category": obj.get("main_category"),
                        "categories": "|".join(obj.get("categories", [])),
                        "average_rating": obj.get("average_rating"),
                        "rating_number": obj.get("rating_number"),
                        "price": obj.get("price"),
                        "date_first_available": details.get("Date First Available")
                    }

                    writer.writerow(row)

        print("Metadata extraction complete.")
    return (extract_from_meta,)


@app.cell
def _(extract_from_meta, top_10_asins, top_15_to_30_asins, top_20_asins):
    extract_from_meta(
        meta_path="datasets/meta_Electronics/meta_Electronics.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/top_10_asins_metadata1.csv",
        target_asins=top_10_asins
    )
    extract_from_meta(
        meta_path="datasets/meta_Home_and_Kitchen/meta_Home_and_Kitchen.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/top_10_asins_metadata1.csv",
        target_asins=top_10_asins,
        mode="a"
    )


    extract_from_meta(
        meta_path="datasets/meta_Electronics/meta_Electronics.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/top_20_asins_metadata1.csv",
        target_asins=top_20_asins
    )
    extract_from_meta(
        meta_path="datasets/meta_Home_and_Kitchen/meta_Home_and_Kitchen.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/top_20_asins_metadata1.csv",
        target_asins=top_20_asins,
        mode="a"
    )

    extract_from_meta(
        meta_path="datasets/meta_Electronics/meta_Electronics.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/asins_15_to_30_metadata1.csv",
        target_asins=top_15_to_30_asins
    )
    extract_from_meta(
        meta_path="datasets/meta_Home_and_Kitchen/meta_Home_and_Kitchen.jsonl",
        output_csv="datasets/filtered_smart_home_reviews/asins_15_to_30_metadata1.csv",
        target_asins=top_15_to_30_asins,
        mode="a"
    )
    return


if __name__ == "__main__":
    app.run()
