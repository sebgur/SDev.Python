from dicttoxml import dicttoxml
import xmltodict
from sdevpy.utilities import jsonmanager as jsm


def serialize(dic: dict, file: str, custom_root: str='') -> None:
    """ Serialize dictionary to xml file """
    if custom_root == '':
        xml = dicttoxml(dic, attr_type=False, return_bytes=False)
    else:
        xml = dicttoxml(dic, attr_type=False, return_bytes=False, custom_root=custom_root)

    with open(file, 'w') as xml_file:
        xml_file.write(xml)


def deserialize(file: str) -> dict:
    """ Deserialize xml file to dictionary """
    with open(file) as xml_file:
        return xmltodict.parse(xml_file.read())


def xml_to_json(xml_file_in: str, json_file_out: str, json_indent: int=2) -> None:
    """ Convert xml file to json """
    # Deserialize xml to object
    obj = deserialize(xml_file_in)

    # Serialize object to json
    jsm.serialize(obj, json_file_out, indent=json_indent)
