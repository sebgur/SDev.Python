""" Json utilities for serializing, deserializing, etc. """
import json
from pydantic import BaseModel
from typing import Optional

def serialize(dic, file):
    """ Serialize dictionary into json file """
    with open(file, "w", encoding='utf8') as write:
        json.dump(dic, write)


def deserialize(file):
    """ Deserialize json file into dictionary """
    with open(file, encoding='utf8') as file_hdl: # 'with' makes it unnecessary to close file_hdl
        dic = json.load(file_hdl)
    return dic


def to_string(dic):
    """ Serialize dictionary into json string """
    jsonstr = json.dumps(dic)
    return jsonstr


def from_string(jsonstr):
    """ Deserialize json string into dictionary """
    dic = json.loads(jsonstr)
    return dic


if __name__ == "__main__":
    # # Raw json
    # DATA = {
    # "user":
    #   {
    #       "name": "seb",
    #       "age": 16,
    #       "place": "Singapore"
    #   }
    # }
    # print(DATA["user"])

    # FILE = r"C:\\temp\\sdevpy\\test.json"
    # JSONSTR = to_string(DATA)
    # print(JSONSTR)

    # NEWDATA = from_string(JSONSTR)
    # NEWDATA["user"]["age"] = 12
    # print(NEWDATA)

    # serialize(NEWDATA, FILE)

    # NEWDATA2 = deserialize(FILE)
    # print(NEWDATA2['user'])

    # Validate using pydantic
    # class Person(BaseModel):
    #     name: str
    #     age: int
    #     email: str
    #     phone: Optional[str]

