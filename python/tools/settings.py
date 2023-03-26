import os

# Disable debug warnings in tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Doing nothing for now, just to avoid warning of dummy impot
def apply_settings():
    return 0
