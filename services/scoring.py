def personalization_weight(item, pref) -> float:
    try:
        category_counter = pref.get("category", {})
        season_counter = pref.get("season", {})

        category_score = category_counter.get(item.mainCategory, 0)
        season_score = season_counter.get(item.season, 0)

        max_category = max(category_counter.values(), default=1)
        max_season = max(season_counter.values(), default=1)

        return (
            0.6 * (category_score / max_category) +
            0.4 * (season_score / max_season)
        )
    except Exception as e:
        print("SCORING ERROR:", e)
        return 0.0
