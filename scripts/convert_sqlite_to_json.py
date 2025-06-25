import sqlite3, json, gzip
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Convert SQLite translations to JSON")
    parser.add_argument("--db", type=str, required=True, help="Path to the SQLite database")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    # conn = sqlite3.connect("translations.db")
    cursor = conn.cursor()
    with open(args.output, "wt") as f:
        for row in cursor.execute("SELECT * FROM translations"):
            json.dump({"location": row[0], "source": row[1], "output": row[2]}, f, ensure_ascii=False)
            f.write("\n")
    conn.close()
