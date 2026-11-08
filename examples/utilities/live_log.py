import csv
import datetime as dt
from pathlib import Path
from sdevpy.utilities import sessions as ssn


CENTERS = ["NEWYORK", "SYDNEY", "SINGAPORE"]


def main(output: Path) -> None:
    now = dt.datetime.now(dt.UTC)
    row = [now.astimezone().strftime("%Y-%m-%d %H:%M")]
    for name in CENTERS:
        center = ssn.make_center(name)
        row += [center.session_date(now).strftime("%Y-%m-%d"), center.is_live(now, now)]

    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            header = ["run_local"]
            for name in CENTERS:
                header += [f"{name}_session", f"{name}_live"]
            writer.writerow(header)
        writer.writerow(row)


if __name__ == "__main__":
    path = Path.home() / "temp" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    file = path / "live_log.tsv"
    main(file)
