""" Live-session logger: run periodically (e.g. every 15 min for a week) to check sdevpy sessions.
    Each run appends one line to a TSV file: the run time, then for each center its current
    session date and whether it is live.
    Usage: python live_log.py [output.tsv]   (default: live_log.tsv next to this script) """
import csv
import datetime as dt
import sys
from pathlib import Path
from sdevpy.utilities import sessions as ssn

CENTERS = list(ssn.CENTER_SPECS)  # Or a subset, e.g. ["NEWYORK", "LONDON", "TOKYO"]


def main(output: Path) -> None:
    now = dt.datetime.now(dt.UTC)
    row = [now.strftime("%Y-%m-%d %H:%M:%S"), now.astimezone().strftime("%a %Y-%m-%d %H:%M %Z")]
    for name in CENTERS:
        center = ssn.make_center(name)
        row += [center.session_date(now).strftime("%a %Y-%m-%d"), center.is_live(now, now)]

    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            header = ["run_utc", "run_machine_local"]
            for name in CENTERS:
                header += [f"{name}_session", f"{name}_live"]
            writer.writerow(header)
        writer.writerow(row)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("live_log.tsv")
    main(out)
