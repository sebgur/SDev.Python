import gc
import pytest
from unittest.mock import MagicMock, patch
from sdevpy.llms.local_model import LocalModel
from sdevpy.llms import huggingface


class ConcreteModel(LocalModel):
    """Minimal concrete subclass for testing non-abstract LocalModel methods."""
    def respond_prompt(self, prompt, **kwargs): return ""
    def chat(self, messages, **kwargs): return ""
    def load(self, max_context_tokens=None): pass
    def unload(self): pass
    def pretty_print(self): pass


class TestLocalModelCleanResponse:
    def setup_method(self):
        self.m = ConcreteModel({})

    def test_no_transformation_needed(self):
        assert self.m.clean_response("hello") == "hello"

    def test_strips_think_block(self):
        assert self.m.clean_response("thinking</think>  actual answer  ") == "actual answer"

    def test_strips_code_fence(self):
        assert self.m.clean_response("```\nprint('hi')\n```") == "print('hi')"

    def test_strips_code_fence_with_language(self):
        assert self.m.clean_response("```python\nprint('hi')\n```") == "print('hi')"

    def test_strips_think_then_code_fence(self):
        assert self.m.clean_response("thinking</think>\n```python\ncode\n```") == "code"

    def test_empty_string(self):
        assert self.m.clean_response("") == ""


class TestLocalModelRespondInstruction:
    def test_calls_chat_with_correct_messages(self):
        m = ConcreteModel({})
        m.chat = MagicMock(return_value="answer")
        result = m.respond_instruction("sys", "user msg")
        m.chat.assert_called_once_with(
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "user msg"}]
        )
        assert result == "answer"

    def test_passes_kwargs_to_chat(self):
        m = ConcreteModel({})
        m.chat = MagicMock(return_value="ok")
        m.respond_instruction("s", "u", max_tokens=512)
        m.chat.assert_called_once_with(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            max_tokens=512,
        )


class TestLocalModelUnderlyingModel:
    def test_returns_model_attribute(self):
        m = ConcreteModel({})
        m.model = "sentinel"
        assert m.underlying_model() == "sentinel"


####### Hugging Face ##############################################################################

def _make_repo(repo_id, size_bytes):
    r = MagicMock()
    r.repo_id = repo_id
    r.size_on_disk = size_bytes
    return r


class TestListAvailableModels:
    @patch("sdevpy.llms.huggingface.scan_cache_dir")
    def test_sorted_by_size_descending(self, mock_scan):
        mock_scan.return_value.repos = [
            _make_repo("small/model", 1 * 1024**3),
            _make_repo("large/model", 4 * 1024**3),
        ]
        models = huggingface.list_available_models()
        assert models[0]["repo_id"] == "large/model"
        assert models[1]["repo_id"] == "small/model"

    @patch("sdevpy.llms.huggingface.scan_cache_dir")
    def test_adds_size_gb_and_size_string(self, mock_scan):
        mock_scan.return_value.repos = [_make_repo("m/m", 2 * 1024**3)]
        models = huggingface.list_available_models()
        assert models[0]["size_gb"] == pytest.approx(2.0)
        assert models[0]["size_string"] == "2.00"

    @patch("sdevpy.llms.huggingface.scan_cache_dir")
    def test_empty_cache_returns_empty_list(self, mock_scan):
        mock_scan.return_value.repos = []
        assert huggingface.list_available_models() == []


class TestModelLocation:
    @patch("sdevpy.llms.huggingface.constants")
    def test_returns_hf_hub_cache(self, mock_constants):
        mock_constants.HF_HUB_CACHE = "/some/cache"
        assert huggingface.model_location() == "/some/cache"


class TestListModelFiles:
    @patch("sdevpy.llms.huggingface.HfApi")
    def test_returns_file_list(self, mock_api_cls):
        mock_api_cls.return_value.list_repo_files.return_value = iter(["f1.gguf", "f2.gguf"])
        files = huggingface.list_model_files("org/repo")
        assert files == ["f1.gguf", "f2.gguf"]


# ── transformers_model ───────────────────────────────────────────────────────

from sdevpy.llms.transformers_model import TransformersModel


def _make_transformers(decoded="decoded output"):
    m = TransformersModel({"repo_id": "x/y"})
    mock_tok = MagicMock()
    mock_inputs = {"input_ids": MagicMock()}
    mock_tok.return_value.to.return_value = mock_inputs
    mock_tok.decode.return_value = decoded
    mock_tok.eos_token_id = 0
    m.tokenizer = mock_tok
    m.model = MagicMock()
    m.model.device = "cpu"
    return m


class TestTransformersModelRespondPrompt:
    def test_returns_clean_response(self):
        m = _make_transformers("plain answer")
        assert m.respond_prompt("hello") == "plain answer"

    def test_default_generation_params(self):
        m = _make_transformers()
        m.respond_prompt("hi")
        _, kwargs = m.model.generate.call_args
        assert kwargs["max_new_tokens"] == 4096
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.9
        assert kwargs["num_beams"] == 1

    def test_kwargs_override_defaults(self):
        m = _make_transformers()
        m.respond_prompt("hi", max_tokens=128, temperature=0.9)
        _, kwargs = m.model.generate.call_args
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["temperature"] == 0.9


class TestTransformersModelChat:
    def test_raises_if_no_chat_template(self):
        m = TransformersModel({"repo_id": "x/y"})
        m.tokenizer = MagicMock()
        m.tokenizer.chat_template = None
        with pytest.raises(ValueError, match="chat template"):
            m.chat([{"role": "user", "content": "hi"}])

    def test_returns_decoded_response(self):
        m = TransformersModel({"repo_id": "x/y"})
        mock_tok = MagicMock()
        mock_tok.chat_template = "tmpl"
        mock_tok.apply_chat_template.return_value = "formatted_prompt"
        mock_inputs = {"input_ids": MagicMock()}
        mock_tok.return_value.to.return_value = mock_inputs
        mock_tok.decode.return_value = "answer"
        mock_tok.eos_token_id = 0
        m.tokenizer = mock_tok
        m.model = MagicMock()
        m.model.device = "cpu"
        assert m.chat([{"role": "user", "content": "hi"}]) == "answer"


class TestTransformersModelLoad:
    @patch("sdevpy.llms.transformers_model.AutoTokenizer")
    @patch("sdevpy.llms.transformers_model.AutoModelForCausalLM")
    @patch("sdevpy.llms.transformers_model.AutoConfig")
    def test_loads_model_and_tokenizer(self, mock_cfg, mock_model_cls, mock_tok_cls):
        m = TransformersModel({"repo_id": "x/y"})
        m.load()
        mock_cfg.from_pretrained.assert_called_once_with("x/y")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tok_cls.from_pretrained.assert_called_once_with("x/y")

    @patch("sdevpy.llms.transformers_model.AutoTokenizer")
    @patch("sdevpy.llms.transformers_model.AutoModelForCausalLM")
    @patch("sdevpy.llms.transformers_model.AutoConfig")
    def test_sets_max_position_embeddings(self, mock_cfg, mock_model_cls, mock_tok_cls):
        cfg_obj = MagicMock()
        mock_cfg.from_pretrained.return_value = cfg_obj
        m = TransformersModel({"repo_id": "x/y"})
        m.load(max_context_tokens=2048)
        assert cfg_obj.max_position_embeddings == 2048


class TestTransformersModelUnload:
    @patch("sdevpy.llms.transformers_model.gc")
    @patch("sdevpy.llms.transformers_model.torch")
    def test_calls_gc_collect(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = False
        m = TransformersModel({})
        m.model = MagicMock()
        m.tokenizer = MagicMock()
        m.unload()
        mock_gc.collect.assert_called_once()

    @patch("sdevpy.llms.transformers_model.gc")
    @patch("sdevpy.llms.transformers_model.torch")
    def test_clears_cuda_cache_when_available(self, mock_torch, mock_gc):
        mock_torch.cuda.is_available.return_value = True
        m = TransformersModel({})
        m.model = MagicMock()
        m.tokenizer = MagicMock()
        m.unload()
        mock_torch.cuda.empty_cache.assert_called_once()


# ── llama_model ──────────────────────────────────────────────────────────────

from sdevpy.llms.llama_model import LlamaModel, get_model


def _make_llama(response_text="llama answer"):
    m = LlamaModel({"repo_id": "org/repo", "filename": "model.gguf"})
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response_text}}]
    }
    m.model = mock_llm
    return m


class TestLlamaModelRespondPrompt:
    def test_returns_response(self):
        assert _make_llama("sky is blue").respond_prompt("Why?") == "sky is blue"

    def test_no_think_appended_by_default(self):
        m = _make_llama()
        m.respond_prompt("question")
        sent = m.model.create_chat_completion.call_args[1]["messages"]
        assert sent[-1]["content"].endswith("\n/no_think")

    def test_no_think_not_appended_when_thinking_enabled(self):
        m = _make_llama()
        m.respond_prompt("question", enable_thinking=True)
        sent = m.model.create_chat_completion.call_args[1]["messages"]
        assert not sent[-1]["content"].endswith("\n/no_think")

    def test_max_tokens_none_becomes_minus_one(self):
        m = _make_llama()
        m.respond_prompt("q", max_tokens=None)
        assert m.model.create_chat_completion.call_args[1]["max_tokens"] == -1

    def test_default_max_tokens_is_4096(self):
        m = _make_llama()
        m.respond_prompt("q")
        assert m.model.create_chat_completion.call_args[1]["max_tokens"] == 4096


class TestLlamaModelChat:
    def test_returns_response(self):
        m = _make_llama("chat answer")
        assert m.chat([{"role": "user", "content": "hello"}]) == "chat answer"

    def test_no_think_appended_to_last_user_message(self):
        m = _make_llama()
        m.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
        sent = m.model.create_chat_completion.call_args[1]["messages"]
        assert sent[-1]["content"].endswith("\n/no_think")

    def test_original_messages_not_mutated(self):
        m = _make_llama()
        original = [{"role": "user", "content": "hi"}]
        m.chat(original)
        assert original[0]["content"] == "hi"

    def test_max_tokens_none_becomes_minus_one(self):
        m = _make_llama()
        m.chat([{"role": "user", "content": "q"}], max_tokens=None)
        assert m.model.create_chat_completion.call_args[1]["max_tokens"] == -1


class TestLlamaModelLoad:
    @patch("sdevpy.llms.llama_model.Llama")
    def test_calls_from_pretrained_with_defaults(self, mock_llama_cls):
        m = LlamaModel({"repo_id": "org/repo", "filename": "m.gguf"})
        m.load()
        mock_llama_cls.from_pretrained.assert_called_once_with(
            repo_id="org/repo", filename="m.gguf", n_ctx=0, verbose=False
        )

    @patch("sdevpy.llms.llama_model.Llama")
    def test_max_context_tokens_passed_as_n_ctx(self, mock_llama_cls):
        m = LlamaModel({"repo_id": "org/repo", "filename": "m.gguf"})
        m.load(max_context_tokens=4096)
        mock_llama_cls.from_pretrained.assert_called_once_with(
            repo_id="org/repo", filename="m.gguf", n_ctx=4096, verbose=False
        )


class TestLlamaModelUnload:
    def test_calls_gc_collect(self):
        m = LlamaModel({})
        m.model = MagicMock()
        with patch("sdevpy.llms.llama_model.gc") as mock_gc:
            m.unload()
            mock_gc.collect.assert_called_once()


class TestGetModel:
    @patch("sdevpy.llms.llama_model.Llama")
    def test_returns_llama_instance(self, mock_llama_cls):
        sentinel = MagicMock()
        mock_llama_cls.from_pretrained.return_value = sentinel
        result = get_model("org/repo", "model.gguf", n_ctx=512)
        assert result is sentinel
        mock_llama_cls.from_pretrained.assert_called_once_with(
            repo_id="org/repo", filename="model.gguf", n_ctx=512, verbose=False
        )


# ── llmfactory ───────────────────────────────────────────────────────────────

from sdevpy.llms import llmfactory


class TestReadLlmConfig:
    def test_reads_config_and_returns_dict(self):
        config = llmfactory.read_llm_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            llmfactory.read_llm_config(folder=str(tmp_path))


class TestGetLlmConfig:
    def test_returns_config_for_valid_id(self):
        config = llmfactory.get_llm_config("tiny-gpt2")
        assert config["type"] == "transformers"
        assert "repo_id" in config

    def test_raises_for_unknown_id(self):
        with pytest.raises(ValueError, match="No model config"):
            llmfactory.get_llm_config("does-not-exist")


class TestListModels:
    @patch("sdevpy.llms.llmfactory.huggingface.list_available_models", return_value=[])
    def test_returns_list_of_strings(self, _):
        models = llmfactory.list_models()
        assert isinstance(models, list)
        assert all(isinstance(n, str) for n in models)

    @patch("sdevpy.llms.llmfactory.huggingface.list_available_models", return_value=[])
    def test_contains_configured_model(self, _):
        assert "tiny-gpt2" in llmfactory.list_models()


class TestFromPretrained:
    @patch("sdevpy.llms.llmfactory.TransformersModel")
    def test_creates_transformers_model(self, mock_cls):
        instance = MagicMock()
        mock_cls.return_value = instance
        result = llmfactory.from_pretrained("tiny-gpt2")
        mock_cls.assert_called_once()
        instance.load.assert_called_once_with(None)
        assert result is instance

    @patch("sdevpy.llms.llmfactory.LlamaModel")
    def test_creates_llama_model(self, mock_cls):
        instance = MagicMock()
        mock_cls.return_value = instance
        result = llmfactory.from_pretrained("qwen3.5-0.8B")
        mock_cls.assert_called_once()
        instance.load.assert_called_once_with(None)
        assert result is instance

    @patch("sdevpy.llms.llmfactory.LlamaModel")
    def test_passes_max_context_tokens_to_load(self, mock_cls):
        instance = MagicMock()
        mock_cls.return_value = instance
        llmfactory.from_pretrained("qwen3.5-0.8B", max_context_tokens=8192)
        instance.load.assert_called_once_with(8192)

    def test_raises_for_unknown_model_type(self):
        bad_config = {"type": "unknown_type", "repo_id": "x/y"}
        with patch("sdevpy.llms.llmfactory.get_llm_config", return_value=bad_config):
            with pytest.raises(ValueError, match="Unknown model type"):
                llmfactory.from_pretrained("anything")


class TestRunInstruction:
    def test_loads_responds_and_unloads(self):
        mock_model = MagicMock()
        mock_model.respond_instruction.return_value = "result"
        with patch("sdevpy.llms.llmfactory.from_pretrained", return_value=mock_model):
            response = llmfactory.run_instruction("tiny-gpt2", "sys", "usr")
        mock_model.respond_instruction.assert_called_once_with("sys", "usr")
        mock_model.unload.assert_called_once()
        assert response == "result"

    def test_unloads_even_on_exception(self):
        mock_model = MagicMock()
        mock_model.respond_instruction.side_effect = RuntimeError("boom")
        with patch("sdevpy.llms.llmfactory.from_pretrained", return_value=mock_model):
            with pytest.raises(RuntimeError):
                llmfactory.run_instruction("tiny-gpt2", "sys", "usr")
        mock_model.unload.assert_called_once()
