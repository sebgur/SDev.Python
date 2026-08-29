

if __name__ == "__main__":
    import os
    import datetime as dt
    from sdevpy.utilities import filemanager
    from sdevpy.utilities import tools
    from openbb import obb
    obb.user.preferences.output_type = "dataframe"

    db_root = tools.workpath() / "database"
    end = dt.date.today()

    # Collect names
    ext = '.tsv'
    all_files = filemanager.list_files(db_root, extensions=[ext])
    names = [f.replace(ext, "") for f in all_files]

    print(f"Found names: {len(names)}")
    print(names)

    # for name in names:
    name ='^SPX'
    df = obb.equity.price.historical(name, provider="yfinance", start_date=dt.date(2024, 5, 1),
                                        end_date=dt.date(2024, 5, 25))
    df = df[['close']]
    print(df.head())
    file = os.path.join(db_root, name + ".tsv")
    df.to_csv(file, index=False, sep='\t')
