# Data management

- Only public posts are analysed. Raw texts live only on the authors' local machines (`data/raw`, `data/interim`, ignored by git).
- The repository contains post IDs, manual codes, word counts and derived tables; usernames are replaced by anonymous author IDs.
- No quotes that could identify an author are published.
- Re-collection: `data/ids/post_ids.csv` + a local Reddit API key in `.env` (never committed). Deleted posts cannot be recovered; their number is logged.
