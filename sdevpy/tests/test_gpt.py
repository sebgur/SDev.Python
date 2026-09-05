from sdevpy.tests import conftest as tst
from sdevpy.llms.gpt_model import GptModel


def test_gpt_training():
    test_path = tst.calibdata_path() / "gpt" / "gpt2-test"
    repo_config = {"type": "gpt", "name": "gpt2-test", "path": test_path}
    # print(test_path)

    data_file = tst.dataset_path() / "llms" / "the-verdict.txt"
    with open(data_file, encoding="utf-8") as f:
        text_data = f.read()

    # print(text_data)

    model = GptModel(repo_config)
    model.load()

    model.train(text_data)

    # print("Testing pretrained")
    response = model.respond_prompt("Why is the sky green?")
    model.unload()

    # print(response)
    test = response[:15]
    ref = "was his--.\n the"
    # print("Test")
    # print(test)
    # print(ref)
    # print(len(test))
    # print(len(ref))
    # for t, r in zip(test, ref):
    #     print(f"{t}/{r}")
    assert test == ref


if __name__ == "__main__":
    test_gpt_training()
